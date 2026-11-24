//! Keeping the keyword index in step with metadata: enable, load, catch
//! up, persist and compaction (ADR 0015)

use super::tokenize::field_text;
use super::{file, InvertedIndex, Slot};
use crate::error::{Error, Result};
use crate::{SrvDB, VectorEngine};

/// Longest accepted field name, in bytes
const MAX_FIELD_BYTES: usize = 256;

pub(super) fn validate_field(field: &str) -> Result<()> {
    if field.is_empty() || field.len() > MAX_FIELD_BYTES || field.split('.').any(str::is_empty) {
        return Err(Error::invalid_input(format!(
            "the keyword field must be 1 to {MAX_FIELD_BYTES} bytes of dot-separated, \
             non-empty names (for example \"text\" or \"doc.body\")"
        )));
    }
    Ok(())
}

impl SrvDB {
    /// Index the string field `field` of every record's metadata for
    /// keyword search (BM25), and keep it indexed as records are added,
    /// deleted and compacted. `"a.b"` names a nested field; an array of
    /// strings is indexed as their concatenation; records without a string
    /// value are not indexed.
    ///
    /// Reuses `keyword.index` if it exists, is valid and covers the same
    /// field; otherwise (no file, another field, a damaged file) builds the
    /// index from metadata, which reads every record once. The index file is
    /// written by the next `persist()`; until then a crash loses only the
    /// index, which this call rebuilds.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for an empty or malformed field name, or an
    /// error reading metadata.
    pub fn enable_keyword_index(&mut self, field: &str) -> Result<()> {
        validate_field(field)?;
        let existing = match file::load(&self.path).map_err(Error::from) {
            Ok(index) => index.filter(|ix| ix.field == field && ix.covered <= self.count()),
            // Derived data: a damaged file is replaced by the rebuild below.
            // Anything else (I/O) is returned to the caller.
            Err(e @ (Error::Corrupt(_) | Error::ChecksumMismatch(_))) => {
                log::warn!("{} is damaged, rebuilding it: {e}", file::FILE_NAME);
                None
            }
            Err(e) => return Err(e),
        };
        let mut index = existing.unwrap_or_else(|| InvertedIndex::new(field));
        self.catch_up(&mut index, true)?;
        *self.keyword.0.get_mut() = Slot::On { index, dirty: true };
        Ok(())
    }

    /// The metadata field indexed for keyword search, or `None` if keyword
    /// search is not enabled.
    ///
    /// # Errors
    /// As [`SrvDB::keyword_search`] when `keyword.index` is read.
    pub fn keyword_field(&self) -> Result<Option<String>> {
        self.with_keyword_index(|ix| ix.map(|ix| ix.field.clone()))
    }

    /// Estimated resident bytes of the keyword index as loaded now
    /// (ADR 0018); 0 while it is off or not read yet
    pub(crate) fn keyword_resident_bytes(&self) -> u64 {
        match &*self.keyword.0.lock() {
            Slot::On { index, .. } => (index.pairs + index.docs() as u64)
                .saturating_mul(crate::tuning::budget::KEYWORD_BYTES),
            _ => 0,
        }
    }

    /// Run `f` on the up-to-date index (`None` when keyword search is off)
    pub(super) fn with_keyword_index<R>(
        &self,
        f: impl FnOnce(Option<&InvertedIndex>) -> R,
    ) -> Result<R> {
        let mut slot = self.keyword.0.lock();
        if matches!(*slot, Slot::Unread) {
            *slot = match file::load(&self.path)? {
                None => Slot::Off,
                Some(mut index) => {
                    // The file may predate deletes: apply all of them once;
                    // later deletes are applied by note_keyword_delete
                    let dirty = self.catch_up(&mut index, true)?;
                    Slot::On { index, dirty }
                }
            };
        }
        match &mut *slot {
            Slot::On { index, dirty } => {
                if index.covered > self.count() {
                    // More records than the database holds (after a repair
                    // truncated it): start again from metadata
                    log::warn!("{} is ahead of the database, rebuilding", file::FILE_NAME);
                    *index = InvertedIndex::new(&index.field);
                }
                *dirty |= self.catch_up(index, false)?;
                Ok(f(Some(index)))
            }
            _ => Ok(f(None)),
        }
    }

    /// Index records added since `index.covered`; with `all_deletes`, also
    /// drop every deleted record (done once, when the index is loaded or
    /// built; afterwards `note_keyword_delete` removes each delete).
    /// Returns whether anything changed.
    fn catch_up(&self, index: &mut InvertedIndex, all_deletes: bool) -> Result<bool> {
        let mut changed = false;
        let count = self.count();
        if index.covered < count {
            let ids: Vec<u64> = (index.covered..count)
                .filter(|id| !self.deleted.contains(id))
                .collect();
            let metas = self.metadata_store.get_batch(&ids)?;
            for (id, meta) in ids.iter().zip(metas) {
                if let Some(text) = meta.and_then(|m| field_text(&m, &index.field)) {
                    index.add(*id, &text);
                }
            }
            log::debug!("keyword index: considered {} new records", ids.len());
            index.covered = count;
            changed = true;
        }
        if all_deletes {
            for id in &self.deleted {
                changed |= index.remove(*id);
            }
        }
        Ok(changed)
    }

    /// `delete()` step: drop `id` from the keyword index if it is loaded.
    /// An index not loaded yet applies every delete when it is loaded.
    pub(crate) fn note_keyword_delete(&mut self, id: u64) {
        if let Slot::On { index, dirty } = self.keyword.0.get_mut() {
            *dirty |= index.remove(id);
        }
    }

    /// `persist()` step: write `keyword.index` if it changed. Runs after
    /// the metadata commit point, so the file never leads metadata.
    pub(crate) fn persist_keyword_index(&mut self) -> Result<()> {
        let mut slot = std::mem::take(self.keyword.0.get_mut());
        let result = match &mut slot {
            Slot::On { index, dirty } => match self.catch_up(index, false) {
                Ok(changed) if changed || *dirty => file::save(&self.path, index)
                    .map(|()| *dirty = false)
                    .map_err(Error::from),
                Ok(_) => Ok(()),
                Err(e) => Err(e),
            },
            _ => Ok(()),
        };
        *self.keyword.0.get_mut() = slot;
        result
    }

    /// `compact()` step, before ids are renumbered: replace `keyword.index`
    /// with an empty index for the same field, so a crash during
    /// compaction leads to a rebuild instead of stale ids.
    pub(crate) fn reset_keyword_index_for_compaction(&mut self) -> Result<()> {
        let Some(field) = self.keyword_field()? else {
            return Ok(());
        };
        let index = InvertedIndex::new(&field);
        file::save(&self.path, &index)?;
        *self.keyword.0.get_mut() = Slot::On { index, dirty: true };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_are_validated() {
        for bad in ["", ".", "a.", ".a", "a..b"] {
            assert!(validate_field(bad).is_err(), "{bad:?}");
        }
        assert!(validate_field(&"x".repeat(MAX_FIELD_BYTES + 1)).is_err());
        assert!(validate_field("doc.body").is_ok());
    }
}
