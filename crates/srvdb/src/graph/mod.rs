//! Typed directed links between records (ADR 0016)
//!
//! Links are primary data. They are held in memory (outgoing and incoming
//! adjacency, link types interned as `u16`) and written to `metadata.db` by
//! the next `persist()`, inside the same durable transaction as the commit
//! point.

mod api;
mod expand;
mod query;
mod store;

#[cfg(test)]
mod tests;

pub use expand::{ExpandedResult, Expansion};

use crate::error::{Error, Result};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

/// Longest link type name
const MAX_TYPE_LEN: usize = 64;

/// Bounds on the graph (ADR 0016). Defaults in [`Limits::DEFAULT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    /// Outgoing links per record, all types together
    pub(crate) out_per_record: usize,
    /// Incoming links per record, all types together
    pub(crate) in_per_record: usize,
    /// Links per database (the graph is kept in memory)
    pub(crate) total: usize,
    /// Distinct link type names per database
    pub(crate) types: usize,
}

impl Limits {
    pub(crate) const DEFAULT: Limits = Limits {
        out_per_record: 1024,
        in_per_record: 100_000,
        total: 1_000_000,
        types: 1024,
    };
}

impl Default for Limits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One change to write at the next commit
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Op {
    Add(u64, Arc<str>, u64),
    Remove(u64, Arc<str>, u64),
}

/// Adjacency set: (interned type, other end)
type Edges = BTreeSet<(u16, u64)>;

/// The in-memory link graph plus the changes not yet persisted
#[derive(Debug, Default)]
pub(crate) struct LinkGraph {
    limits: Limits,
    /// Interned type names; a type id indexes this
    names: Vec<Arc<str>>,
    type_ids: HashMap<Arc<str>, u16>,
    /// source -> {(type, target)}
    out: HashMap<u64, Edges>,
    /// target -> {(type, source)}
    inc: HashMap<u64, Edges>,
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

fn degree(map: &HashMap<u64, Edges>, id: u64) -> usize {
    map.get(&id).map_or(0, BTreeSet::len)
}

impl LinkGraph {
    pub(crate) fn with_limits(limits: Limits) -> Self {
        Self {
            limits,
            ..Self::default()
        }
    }

    /// Number of links
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn type_id(&self, name: &str) -> Option<u16> {
        self.type_ids.get(name).copied()
    }

    /// Intern `name`, or describe the limit it would exceed
    fn intern(&mut self, name: &str) -> std::result::Result<u16, String> {
        if let Some(id) = self.type_id(name) {
            return Ok(id);
        }
        if self.names.len() >= self.limits.types {
            return Err(format!(
                "the database already uses {} link types, the most allowed",
                self.limits.types
            ));
        }
        let id = self.names.len() as u16; // < limits.types <= u16::MAX
        let name: Arc<str> = Arc::from(name);
        self.names.push(name.clone());
        self.type_ids.insert(name, id);
        Ok(id)
    }

    /// Add a link in memory without queueing it, enforcing every limit.
    /// `Ok(false)` if it existed; `Err` describes the limit exceeded.
    fn insert_checked(
        &mut self,
        src: u64,
        name: &str,
        dst: u64,
    ) -> std::result::Result<bool, String> {
        if self.contains(src, name, dst) {
            return Ok(false);
        }
        let limits = self.limits;
        if degree(&self.out, src) >= limits.out_per_record {
            return Err(format!(
                "record {src} already has {} outgoing links, the most allowed",
                limits.out_per_record
            ));
        }
        if degree(&self.inc, dst) >= limits.in_per_record {
            return Err(format!(
                "record {dst} already has {} incoming links, the most allowed",
                limits.in_per_record
            ));
        }
        if self.len >= limits.total {
            return Err(format!(
                "the database already holds {} links, the most allowed",
                limits.total
            ));
        }
        let t = self.intern(name)?;
        self.insert_raw(src, t, dst);
        Ok(true)
    }

    /// Insert without limit checks (callers checked them)
    fn insert_raw(&mut self, src: u64, t: u16, dst: u64) -> bool {
        let new = self.out.entry(src).or_default().insert((t, dst));
        if new {
            self.inc.entry(dst).or_default().insert((t, src));
            self.len += 1;
        }
        new
    }

    /// Add a link, enforcing the limits. Returns whether it is new.
    pub(crate) fn add(&mut self, src: u64, name: &str, dst: u64) -> Result<bool> {
        let new = self
            .insert_checked(src, name, dst)
            .map_err(Error::invalid_input)?;
        if new {
            let t = self.names[self.type_ids[name] as usize].clone();
            self.pending.push(Op::Add(src, t, dst));
        }
        Ok(new)
    }

    /// Remove one direction's entry, dropping an emptied set
    fn detach(map: &mut HashMap<u64, Edges>, key: u64, entry: (u16, u64)) {
        if let Some(set) = map.get_mut(&key) {
            set.remove(&entry);
            if set.is_empty() {
                map.remove(&key);
            }
        }
    }

    /// Remove a link. Returns whether it existed.
    pub(crate) fn remove(&mut self, src: u64, name: &str, dst: u64) -> bool {
        let Some(t) = self.type_id(name) else {
            return false;
        };
        if !self.out.get(&src).is_some_and(|s| s.contains(&(t, dst))) {
            return false;
        }
        Self::detach(&mut self.out, src, (t, dst));
        Self::detach(&mut self.inc, dst, (t, src));
        self.len -= 1;
        self.pending
            .push(Op::Remove(src, self.names[t as usize].clone(), dst));
        true
    }

    pub(crate) fn contains(&self, src: u64, name: &str, dst: u64) -> bool {
        self.type_id(name)
            .is_some_and(|t| self.out.get(&src).is_some_and(|s| s.contains(&(t, dst))))
    }

    /// Detach every link of `id` (both directions), queue their removal
    /// and return them as (type, other end) for outgoing and incoming.
    /// Takes `id`'s own sets instead of copying the graph.
    fn take_record(&mut self, id: u64) -> (Edges, Edges) {
        let outs = self.out.remove(&id).unwrap_or_default();
        let ins = self.inc.remove(&id).unwrap_or_default();
        for &(t, dst) in &outs {
            Self::detach(&mut self.inc, dst, (t, id));
            self.pending
                .push(Op::Remove(id, self.names[t as usize].clone(), dst));
        }
        for &(t, src) in &ins {
            Self::detach(&mut self.out, src, (t, id));
            self.pending
                .push(Op::Remove(src, self.names[t as usize].clone(), id));
        }
        self.len -= outs.len() + ins.len();
        (outs, ins)
    }

    /// Remove every link of a deleted record (both directions)
    pub(crate) fn remove_record(&mut self, id: u64) {
        self.take_record(id);
    }

    /// Whether `move_record(from, to)` stays within the limits (checked
    /// with upper bounds: links that would merge are counted twice)
    pub(crate) fn check_move(&self, from: u64, to: u64) -> Result<()> {
        let limits = self.limits;
        if degree(&self.out, to) + degree(&self.out, from) > limits.out_per_record {
            return Err(Error::invalid_input(format!(
                "moving links would give record {to} more than {} outgoing links",
                limits.out_per_record
            )));
        }
        if degree(&self.inc, to) + degree(&self.inc, from) > limits.in_per_record {
            return Err(Error::invalid_input(format!(
                "moving links would give record {to} more than {} incoming links",
                limits.in_per_record
            )));
        }
        Ok(())
    }

    /// Move every link of `from` to `to` (replace keeps relationships).
    /// Links between the two would become self-links and are dropped.
    /// Call `check_move` first; the total never grows.
    pub(crate) fn move_record(&mut self, from: u64, to: u64) {
        let (outs, ins) = self.take_record(from);
        let moved = outs
            .into_iter()
            .map(|(t, dst)| (to, t, dst))
            .chain(ins.into_iter().map(|(t, src)| (src, t, to)));
        for (src, t, dst) in moved {
            if src != dst && self.insert_raw(src, t, dst) {
                self.pending
                    .push(Op::Add(src, self.names[t as usize].clone(), dst));
            }
        }
    }

    pub(crate) fn pending(&self) -> &[Op] {
        &self.pending
    }

    /// Forget the pending changes once a commit wrote them, releasing
    /// their buffer (a bulk link load should not keep it allocated)
    pub(crate) fn clear_pending(&mut self) {
        self.pending = Vec::new();
    }
}
