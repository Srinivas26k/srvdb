//! Delete and replace (ADR 0004)

use super::*;

impl SrvDB {
    /// Delete the vector with internal id `id`.
    ///
    /// The vector is tombstoned: searches skip it and its metadata is
    /// removed, but no other id changes. Like adds, it is durable after
    /// `persist()`. Returns `false` if `id` was already deleted.
    ///
    /// # Errors
    /// `InvalidInput` if `id` was never assigned.
    pub fn delete(&mut self, id: u64) -> Result<bool> {
        if id >= self.count() {
            return Err(Error::invalid_input(format!(
                "id {id} does not exist (the database holds ids 0..{})",
                self.count()
            )));
        }
        if self.deleted.contains(&id) {
            return Ok(false);
        }
        // Takes effect now; written to disk by the next persist(), so an
        // unpersisted delete is undone on reopen exactly like an add
        self.deleted.insert(id);
        self.pending_deletes.push(id);
        self.graph.remove_record(id); // a deleted record has no links (ADR 0016)
        self.note_keyword_delete(id);
        Ok(true)
    }

    /// Replace the vector and metadata of `id`: the old record is deleted
    /// and the new one appended, and its links move to the new id
    /// (ADR 0016). Returns the new id.
    ///
    /// # Errors
    /// `InvalidInput` if `id` does not exist or is already deleted, or if
    /// the new vector or metadata are invalid (checked before anything
    /// changes).
    pub fn replace(&mut self, id: u64, vector: &Vector, meta: &str) -> Result<u64> {
        if id >= self.count() || self.deleted.contains(&id) {
            return Err(Error::invalid_input(format!(
                "id {id} does not exist or was deleted"
            )));
        }
        validate::vector(&vector.data, self.dimension)?;
        validate::metadata(meta)?;
        // Checked before anything is written, so an error leaves no
        // orphaned new record; the new id has no links yet
        self.graph.check_move(id, self.count())?;
        let new_id = self.insert_one(vector, meta)?;
        // Cannot exceed a limit: check_move passed for this empty target
        // and the total never grows (links follow the record, ADR 0016)
        self.graph.move_record(id, new_id);
        self.delete(id)?;
        Ok(new_id)
    }

    /// Whether `id` has been deleted
    pub fn is_deleted(&self, id: u64) -> bool {
        self.deleted.contains(&id)
    }

    /// Number of searchable (not deleted) vectors. `count()` is the number
    /// of stored records, deleted ones included.
    pub fn live_count(&self) -> u64 {
        self.count() - self.deleted.len() as u64
    }
}
