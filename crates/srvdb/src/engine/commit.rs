//! Persistence: the commit protocol from ADR 0003

use super::*;

impl SrvDB {
    pub(super) fn commit(&mut self) -> Result<()> {
        // Auto-Tuning Hook: Check if we should upgrade strategy
        if let Err(e) = auto_tune::check_and_migrate(self) {
            // Migration is an optimisation; the data is already safe without it
            log::warn!("auto-tune migration skipped: {e}");
        }

        // 1. Vectors on disk (fsync)
        if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.flush()?;
        }
        if let Some(ref mut qstorage) = self.quantized_storage {
            qstorage.flush()?;
        }
        if let Some(ref mut scalar) = self.scalar_storage {
            scalar.flush()?;
        }

        // 2. The commit point (ADR 0003): metadata, the count and links
        //    (ADR 0016) are durable
        let count = self.count();
        self.commit_with_links(count)?;
        self.pending_deletes.clear();

        // 3. Derived indexes, each replaced atomically
        if let Some(ref mut codes) = self.rabitq {
            codes.flush()?;
        }
        if let Some(ref ivf) = self.ivf_index {
            storage::ivf::IVFStorage::save(ivf, &self.path)?;
        }
        if let Some(ref hnsw) = self.hnsw_index {
            storage::atomic::write_atomic(&self.path.join("hnsw.graph"), &hnsw.to_bytes()?)?;
        }

        // 4. Index settings (ADR 0011), replaced atomically if they changed
        // (set_mode, configure_ivf, set_ef_search, ...). Only updated, never
        // created here: a database without them gets them from an explicit
        // open, so a legacy database opened flat by default is not pinned
        // to flat.
        if self.stored_settings.is_some() {
            self.record_settings()?;
        }

        Ok(())
    }

    /// All stored (id, metadata) pairs
    pub fn all_metadata(&self) -> Result<Vec<(u64, String)>> {
        let mut all = self.metadata_store.all()?;
        all.retain(|(id, _)| !self.deleted.contains(id));
        Ok(all)
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> Option<quantized_storage::StorageStats> {
        self.quantized_storage.as_ref().map(|s| s.get_stats())
    }
}
