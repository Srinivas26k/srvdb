//! `SrvDB` link methods and the engine hooks that keep links in step with
//! commit, recovery and compaction (ADR 0016)

use super::expand::{expand, ExpandedResult, Expansion};
use super::{store, validate_type, LinkGraph};
use crate::error::{Error, Result};
use crate::{SrvDB, Vector, VectorEngine};

impl SrvDB {
    /// Check that `id` is an existing, not deleted record
    fn live_record(&self, id: u64, role: &str) -> Result<()> {
        if id >= self.count() || self.deleted.contains(&id) {
            return Err(Error::invalid_input(format!(
                "{role} id {id} does not exist or was deleted"
            )));
        }
        Ok(())
    }

    /// Add the directed link `source -link_type-> target`.
    ///
    /// Takes effect at once (seen by [`SrvDB::neighbors`] and
    /// [`SrvDB::search_expand`]) and is durable after the next `persist()`,
    /// atomically with the records committed by it. Returns `false` if the
    /// link already existed.
    ///
    /// # Errors
    /// `InvalidInput` if either id does not exist or was deleted, if
    /// `source == target`, if `link_type` does not match
    /// `[a-z][a-z0-9_]{0,63}`, or if `source` already has 1,024 outgoing
    /// links or the database 10,000,000 links.
    pub fn link(&mut self, source: u64, link_type: &str, target: u64) -> Result<bool> {
        validate_type(link_type)?;
        self.live_record(source, "source")?;
        self.live_record(target, "target")?;
        if source == target {
            return Err(Error::invalid_input(format!(
                "a record cannot link to itself (id {source})"
            )));
        }
        self.graph.add(source, link_type, target)
    }

    /// Remove the link `source -link_type-> target`. Returns `false` if it
    /// did not exist. Durable after the next `persist()`.
    ///
    /// # Errors
    /// `InvalidInput` if `link_type` is not a valid link type name.
    pub fn unlink(&mut self, source: u64, link_type: &str, target: u64) -> Result<bool> {
        validate_type(link_type)?;
        Ok(self.graph.remove(source, link_type, target))
    }

    /// Outgoing links of `id` as `(link type, target id)`, sorted, only of
    /// `link_type` if given. Empty for a record without links.
    ///
    /// # Errors
    /// `InvalidInput` if `id` does not exist or was deleted, or `link_type`
    /// is not a valid link type name.
    pub fn neighbors(&self, id: u64, link_type: Option<&str>) -> Result<Vec<(String, u64)>> {
        if let Some(t) = link_type {
            validate_type(t)?;
        }
        self.live_record(id, "record")?;
        Ok(self.graph.outgoing(id, link_type))
    }

    /// Total number of links
    pub fn link_count(&self) -> u64 {
        self.graph.len() as u64
    }

    /// Move every link of `from`, in both directions, to `to`, as
    /// [`SrvDB::replace`] does. A link between the two is dropped. Use it
    /// when replacing a record by adding a new one and deleting the old.
    /// Durable after the next `persist()`.
    ///
    /// # Errors
    /// `InvalidInput` if either id does not exist or was deleted, or `to`
    /// would get more than 1,024 outgoing links.
    pub fn move_links(&mut self, from: u64, to: u64) -> Result<()> {
        self.live_record(from, "source")?;
        self.live_record(to, "target")?;
        if from != to {
            self.graph.move_record(from, to)?;
        }
        Ok(())
    }

    /// Search for the `k` nearest records, then follow outgoing links of
    /// `expansion.link_type` breadth-first for up to `expansion.hops` hops.
    ///
    /// Returns the hits (hop 0, best first) followed by at most
    /// `expansion.max_results` linked records, ordered by hop and then by
    /// the score they inherit from the record they were reached from. Each
    /// record appears once, at its smallest hop. Thread-safe like `search`.
    ///
    /// # Errors
    /// As for `search`, plus `InvalidInput` if `hops` is not 1 to 3,
    /// `max_results` not 1 to 1,000 or the link type is invalid.
    pub fn search_expand(
        &self,
        query: &Vector,
        k: usize,
        expansion: &Expansion,
    ) -> Result<Vec<ExpandedResult>> {
        expansion.validate()?;
        let hits = self.search(query, k)?;
        let expanded = expand(&self.graph, &hits, expansion);
        let ids: Vec<u64> = expanded.iter().map(|&(id, _, _)| id).collect();
        let metadata = self.metadata_store.get_batch(&ids)?;
        Ok(expanded
            .into_iter()
            .zip(metadata)
            .map(|((id, score, hop), metadata)| ExpandedResult {
                id,
                score,
                hop,
                metadata,
            })
            .collect())
    }

    /// Recovery hook: load and validate the stored links
    pub(crate) fn load_links(&mut self) -> Result<()> {
        self.graph = store::load(self.metadata_store.database(), self.count(), &self.deleted)?;
        Ok(())
    }

    /// Commit hook: the durable commit, with pending link changes written
    /// in the same transaction
    pub(crate) fn commit_with_links(&mut self, count: u64) -> Result<()> {
        let ops = self.graph.pending();
        self.metadata_store
            .commit_with(count, &self.pending_deletes, |txn| store::apply(txn, ops))?;
        self.graph.clear_pending();
        Ok(())
    }

    /// Compaction hook: renumber metadata and links in one durable
    /// transaction; the in-memory graph follows once it committed
    pub(crate) fn compact_with_links(&mut self, keep: &[u64]) -> Result<()> {
        let renumbered: LinkGraph = self.graph.renumbered(keep);
        self.metadata_store
            .compact_with(keep, |txn| store::rewrite(txn, &renumbered))?;
        self.graph = renumbered;
        Ok(())
    }
}
