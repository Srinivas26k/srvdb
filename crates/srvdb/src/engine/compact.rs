//! Compaction: physically remove deleted vectors (ADR 0004)

use super::*;
use crate::storage::compaction;

impl SrvDB {
    /// Remove deleted vectors from disk and renumber ids densely.
    ///
    /// Persists first, so everything written so far is kept. Internal ids
    /// change: the `i`-th live vector (in id order) becomes id `i`. Python
    /// string ids are unaffected. HNSW and IVF indexes are rebuilt, which
    /// takes time proportional to the data. Crash-safe: an interrupted
    /// compaction is finished or discarded on the next open.
    ///
    /// Returns the number of vectors removed (0 if there was nothing to do).
    pub fn compact(&mut self) -> Result<u64> {
        self.commit()?;
        if self.deleted.is_empty() {
            return Ok(0);
        }
        let removed = self.deleted.len() as u64;
        let keep: Vec<u64> = (0..self.count())
            .filter(|id| !self.deleted.contains(id))
            .collect();

        // 1. Compacted copy of the active vector file, fsynced
        let dir = self.path.clone();
        if let Some(ref s) = self.vector_storage {
            s.write_compacted(&compaction::staged(&dir, storage::flat::FILE_NAME), &keep)?;
        }
        if let Some(ref s) = self.scalar_storage {
            s.write_compacted(&compaction::staged(&dir, storage::sq::FILE_NAME), &keep)?;
        }
        if let Some(ref s) = self.quantized_storage {
            s.write_compacted(&compaction::staged(&dir, storage::pq::FILE_NAME), &keep)?;
        }

        // 2. The commit point: renumbered metadata, flagged as pending
        self.metadata_store.compact(&keep)?;

        // 3. Swap files (old ones must be closed first on Windows), then
        //    reopen and rebuild the derived indexes
        let had_vectors = self.vector_storage.take().is_some();
        let had_sq8 = self.scalar_storage.take().is_some();
        let had_pq = self.quantized_storage.take().is_some();
        compaction::finish(&dir)?;
        self.metadata_store.finish_compaction()?;

        let path = dir
            .to_str()
            .ok_or_else(|| Error::invalid_input("database path is not valid UTF-8"))?;
        if had_vectors {
            self.vector_storage = Some(VectorStorage::new(path, self.dimension)?);
        }
        if had_sq8 {
            self.scalar_storage = Some(ScalarQuantizedStorage::load(path, self.dimension)?);
        }
        if had_pq {
            self.quantized_storage = Some(QuantizedVectorStorage::load(path)?);
        }
        self.deleted.clear();
        self.pending_deletes.clear();
        self.rebuild_indexes()?;
        log::info!("compaction removed {removed} deleted vectors");
        Ok(removed)
    }

    /// Rebuild HNSW and IVF from the vectors after ids changed
    fn rebuild_indexes(&mut self) -> Result<()> {
        if self.hnsw_index.take().is_some() {
            let config = self.hnsw_config.clone().unwrap_or_default();
            self.attach_hnsw(config)?;
        }
        let ivf_trained = self
            .ivf_index
            .as_ref()
            .is_some_and(|ivf| !ivf.centroids.is_empty());
        if ivf_trained {
            self.train_ivf()?;
        }
        self.commit()
    }
}
