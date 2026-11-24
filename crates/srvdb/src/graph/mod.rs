//! Typed directed links between records (ADR 0016)
//!
//! Links are primary data. They are held in memory (outgoing and incoming
//! adjacency) and written to `metadata.db` by the next `persist()`, inside
//! the same durable transaction as the commit point.

mod api;
mod expand;
mod store;

#[cfg(test)]
mod tests;

pub use expand::{ExpandedResult, Expansion};

use crate::error::{Error, Result};
use std::collections::{BTreeSet, HashMap};

/// Most outgoing links one record may have, all types together
pub(crate) const MAX_LINKS_PER_RECORD: usize = 1024;
/// Most links one database may hold (the graph is kept in memory)
pub(crate) const MAX_LINKS: usize = 10_000_000;
/// Longest link type name
const MAX_TYPE_LEN: usize = 64;

/// One change to write at the next commit
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Op {
    Add(u64, String, u64),
    Remove(u64, String, u64),
}

/// The in-memory link graph plus the changes not yet persisted
#[derive(Debug, Default)]
pub(crate) struct LinkGraph {
    /// source -> {(type, target)}
    out: HashMap<u64, BTreeSet<(String, u64)>>,
    /// target -> {(type, source)}
    inc: HashMap<u64, BTreeSet<(String, u64)>>,
    len: usize,
    pending: Vec<Op>,
}

/// Check a link type against the allow-pattern `[a-z][a-z0-9_]{0,63}`
pub(crate) fn validate_type(link_type: &str) -> Result<()> {
    let mut chars = link_type.chars();
    let first_ok = chars.next().is_some_and(|c| c.is_ascii_lowercase());
    let rest_ok = chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if first_ok && rest_ok && link_type.len() <= MAX_TYPE_LEN {
        Ok(())
    } else {
        Err(Error::invalid_input(format!(
            "link type must start with a lowercase letter and contain only \
             lowercase letters, digits and '_', at most {MAX_TYPE_LEN} characters \
             (got {} characters)",
            link_type.len()
        )))
    }
}

impl LinkGraph {
    /// Number of links
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Add a link in memory without queueing it (loading from disk)
    fn insert_loaded(&mut self, src: u64, ty: &str, dst: u64) -> bool {
        let new = self
            .out
            .entry(src)
            .or_default()
            .insert((ty.to_string(), dst));
        if new {
            self.inc
                .entry(dst)
                .or_default()
                .insert((ty.to_string(), src));
            self.len += 1;
        }
        new
    }

    /// Remove a link in memory without queueing it
    fn remove_unqueued(&mut self, src: u64, ty: &str, dst: u64) -> bool {
        let key = (ty.to_string(), dst);
        let removed = self.out.get_mut(&src).is_some_and(|s| s.remove(&key));
        if removed {
            if self.out.get(&src).is_some_and(BTreeSet::is_empty) {
                self.out.remove(&src);
            }
            if let Some(set) = self.inc.get_mut(&dst) {
                set.remove(&(ty.to_string(), src));
                if set.is_empty() {
                    self.inc.remove(&dst);
                }
            }
            self.len -= 1;
        }
        removed
    }

    /// Add a link, enforcing the limits. Returns whether it is new.
    pub(crate) fn add(&mut self, src: u64, ty: &str, dst: u64) -> Result<bool> {
        if self.contains(src, ty, dst) {
            return Ok(false);
        }
        if self.out.get(&src).map_or(0, BTreeSet::len) >= MAX_LINKS_PER_RECORD {
            return Err(Error::invalid_input(format!(
                "record {src} already has {MAX_LINKS_PER_RECORD} outgoing links, the most allowed"
            )));
        }
        if self.len >= MAX_LINKS {
            return Err(Error::invalid_input(format!(
                "the database already holds {MAX_LINKS} links, the most allowed"
            )));
        }
        self.insert_loaded(src, ty, dst);
        self.pending.push(Op::Add(src, ty.to_string(), dst));
        Ok(true)
    }

    /// Remove a link. Returns whether it existed.
    pub(crate) fn remove(&mut self, src: u64, ty: &str, dst: u64) -> bool {
        let removed = self.remove_unqueued(src, ty, dst);
        if removed {
            self.pending.push(Op::Remove(src, ty.to_string(), dst));
        }
        removed
    }

    pub(crate) fn contains(&self, src: u64, ty: &str, dst: u64) -> bool {
        self.out
            .get(&src)
            .is_some_and(|s| s.contains(&(ty.to_string(), dst)))
    }

    /// Every link touching `id`, as (source, type, target)
    fn touching(&self, id: u64) -> Vec<(u64, String, u64)> {
        let out = self.out.get(&id).into_iter().flatten();
        let inc = self.inc.get(&id).into_iter().flatten();
        out.map(|(t, dst)| (id, t.clone(), *dst))
            .chain(inc.map(|(t, src)| (*src, t.clone(), id)))
            .collect()
    }

    /// Remove every link of a deleted record (both directions)
    pub(crate) fn remove_record(&mut self, id: u64) {
        for (src, ty, dst) in self.touching(id) {
            self.remove(src, &ty, dst);
        }
    }

    /// Move every link of `from` to `to` (replace keeps relationships).
    /// Links that would become self-links (between the two) are dropped.
    pub(crate) fn move_record(&mut self, from: u64, to: u64) -> Result<()> {
        let links = self.touching(from);
        let outgoing = links.iter().filter(|(s, _, d)| *s == from && *d != to);
        let extra = outgoing.count();
        if self.out.get(&to).map_or(0, BTreeSet::len) + extra > MAX_LINKS_PER_RECORD {
            return Err(Error::invalid_input(format!(
                "moving links would give record {to} more than \
                 {MAX_LINKS_PER_RECORD} outgoing links"
            )));
        }
        for (src, ty, dst) in links {
            self.remove(src, &ty, dst);
            let (src, dst) = (
                if src == from { to } else { src },
                if dst == from { to } else { dst },
            );
            if src != dst && !self.contains(src, &ty, dst) {
                self.insert_loaded(src, &ty, dst);
                self.pending.push(Op::Add(src, ty, dst));
            }
        }
        Ok(())
    }

    /// Outgoing links of `id`, optionally of one type, sorted
    pub(crate) fn outgoing(&self, id: u64, ty: Option<&str>) -> Vec<(String, u64)> {
        self.out
            .get(&id)
            .into_iter()
            .flatten()
            .filter(|(t, _)| ty.is_none_or(|ty| t == ty))
            .cloned()
            .collect()
    }

    /// Targets of `id`'s outgoing links of type `ty`
    pub(crate) fn targets<'a>(&'a self, id: u64, ty: &'a str) -> impl Iterator<Item = u64> + 'a {
        self.out
            .get(&id)
            .into_iter()
            .flatten()
            .filter(move |(t, _)| t == ty)
            .map(|(_, dst)| *dst)
    }

    /// Changes not yet persisted
    pub(crate) fn pending(&self) -> &[Op] {
        &self.pending
    }

    /// Forget the pending changes once a commit wrote them
    pub(crate) fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Renumber after compaction: `keep[i]` becomes `i`. Links touching
    /// records not in `keep` are dropped (there are none: delete removes
    /// them). Must be called after the pending changes were committed.
    pub(crate) fn renumbered(&self, keep: &[u64]) -> LinkGraph {
        let new_id: HashMap<u64, u64> = keep
            .iter()
            .enumerate()
            .map(|(i, &old)| (old, i as u64))
            .collect();
        let mut graph = LinkGraph::default();
        for (src, links) in &self.out {
            for (ty, dst) in links {
                if let (Some(&s), Some(&d)) = (new_id.get(src), new_id.get(dst)) {
                    graph.insert_loaded(s, ty, d);
                }
            }
        }
        graph
    }

    /// Every link as (source, type, target), sorted
    pub(crate) fn all(&self) -> Vec<(u64, String, u64)> {
        let mut all: Vec<_> = self
            .out
            .iter()
            .flat_map(|(src, set)| set.iter().map(|(t, d)| (*src, t.clone(), *d)))
            .collect();
        all.sort();
        all
    }
}
