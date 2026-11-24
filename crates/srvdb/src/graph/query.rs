//! Reading and renumbering the link graph (ADR 0016)

use super::LinkGraph;
use std::collections::HashMap;
use std::sync::Arc;

impl LinkGraph {
    /// Outgoing links of `id`, optionally of one type, sorted by name
    pub(crate) fn outgoing(&self, id: u64, name: Option<&str>) -> Vec<(String, u64)> {
        let want = name.map(|n| self.type_id(n));
        let mut links: Vec<(String, u64)> = self
            .out
            .get(&id)
            .into_iter()
            .flatten()
            .filter(|(t, _)| want.is_none_or(|w| w == Some(*t)))
            .map(|&(t, dst)| (self.names[t as usize].to_string(), dst))
            .collect();
        links.sort();
        links
    }

    /// Targets of `id`'s outgoing links of type `name`
    pub(crate) fn targets(&self, id: u64, name: &str) -> impl Iterator<Item = u64> + '_ {
        let t = self.type_id(name);
        self.out
            .get(&id)
            .into_iter()
            .flatten()
            .filter(move |(tt, _)| Some(*tt) == t)
            .map(|&(_, dst)| dst)
    }

    /// Renumber after compaction: `keep[i]` becomes `i`. Links touching
    /// records not in `keep` are dropped (there are none: delete removes
    /// them). Call after the pending changes were committed.
    pub(crate) fn renumbered(&self, keep: &[u64]) -> LinkGraph {
        let new_id: HashMap<u64, u64> = keep
            .iter()
            .enumerate()
            .map(|(i, &old)| (old, i as u64))
            .collect();
        let mut graph = LinkGraph {
            limits: self.limits,
            names: self.names.clone(),
            type_ids: self.type_ids.clone(),
            ..Self::default()
        };
        for (src, links) in &self.out {
            for &(t, dst) in links {
                if let (Some(&s), Some(&d)) = (new_id.get(src), new_id.get(&dst)) {
                    graph.insert_raw(s, t, d);
                }
            }
        }
        graph
    }

    /// Every link as (source, type, target), sorted
    pub(crate) fn all(&self) -> Vec<(u64, Arc<str>, u64)> {
        let mut all: Vec<_> = self
            .out
            .iter()
            .flat_map(|(src, set)| {
                set.iter()
                    .map(|&(t, d)| (*src, self.names[t as usize].clone(), d))
            })
            .collect();
        all.sort();
        all
    }
}
