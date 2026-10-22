//! Crash recovery on open (ADR 0003)

use super::*;

impl SrvDB {
    /// Bring storage back to the last durable commit.
    ///
    /// Vectors and metadata written after the last `persist()` are removed.
    /// A database written before v0.3 has no commit record; its vector-file
    /// count is taken as committed and recorded.
    pub(super) fn recover(&mut self) -> Result<()> {
        let stored = self.count();
        let committed = match self.metadata_store.committed_count()? {
            Some(committed) => committed,
            None => {
                self.metadata_store.commit(stored, &[])?;
                stored
            }
        };
        if committed > stored {
            return Err(Error::Corrupt(format!(
                "{committed} vectors were committed but the vector file holds {stored}"
            )));
        }
        if committed < stored {
            log::warn!(
                "discarding {} vectors written after the last persist()",
                stored - committed
            );
        }
        if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.truncate_to(committed)?;
        }
        if let Some(ref mut qstorage) = self.quantized_storage {
            qstorage.truncate_to(committed)?;
        }
        if let Some(ref mut scalar) = self.scalar_storage {
            scalar.truncate_to(committed)?;
        }
        self.metadata_store.truncate_from(committed)?;
        self.deleted = self.metadata_store.tombstones()?.into_iter().collect();
        Ok(())
    }
}
