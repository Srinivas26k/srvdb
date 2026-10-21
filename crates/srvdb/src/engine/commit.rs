//! Persistence: flushing storage and saving indexes

use super::*;

impl SrvDB {
    pub(super) fn commit(&mut self) -> Result<()> {
        // Auto-Tuning Hook: Check if we should upgrade strategy
        if let Err(e) = auto_tune::check_and_migrate(self) {
            eprintln!("Auto-Tuning Migration Warning: {}", e);
        }

        if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.flush()?;
        }
        if let Some(ref mut qstorage) = self.quantized_storage {
            qstorage.flush()?;
        }
        if let Some(ref mut scalar) = self.scalar_storage {
            scalar.flush()?;
        }

        if let Some(ref ivf) = self.ivf_index {
            storage::ivf::IVFStorage::save(ivf, &self.path)?;
        }

        // HNSW Persistence
        if let Some(ref hnsw) = self.hnsw_index {
            let graph_path = self.path.join("hnsw.graph");
            let bytes = hnsw.to_bytes()?;
            std::fs::write(graph_path, bytes)?;
        }

        self.metadata_store.flush()?;
        Ok(())
    }

    /// Make buffered vector writes visible to search.
    ///
    /// Cheap compared to `persist()`: only flushes vector storage (a no-op when
    /// nothing was added) and does not serialize indexes or run auto-tuning.
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.flush()?;
        }
        if let Some(ref mut qstorage) = self.quantized_storage {
            qstorage.flush()?;
        }
        if let Some(ref mut scalar) = self.scalar_storage {
            scalar.flush()?;
        }
        Ok(())
    }

    /// All stored (id, metadata) pairs
    pub fn all_metadata(&self) -> Result<Vec<(u64, String)>> {
        self.metadata_store.all()
    }

    /// Remove the metadata entry for an id
    pub fn delete_metadata(&mut self, id: u64) -> Result<()> {
        self.metadata_store.delete(id)
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> Option<quantized_storage::StorageStats> {
        self.quantized_storage.as_ref().map(|s| s.get_stats())
    }
}
