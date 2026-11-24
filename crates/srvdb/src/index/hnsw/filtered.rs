//! Search that returns only accepted ids (ADR 0004, issue #25)
//!
//! Rejected nodes (deleted ones, or ones failing a metadata filter) stay
//! in the graph and are still walked through for navigation, but they
//! never enter the result set. The layer-0 search keeps exploring until it
//! holds `ef` accepted results, so it returns `k` results whenever `k`
//! accepted nodes are reachable.
//!
//! When few direct neighbours of an expanded node are accepted, the search
//! also scores the accepted neighbours of its rejected neighbours (the
//! two-hop expansion from ACORN, Patel et al. 2024). That keeps the walk
//! close to the accepted subgraph when a filter thins out the graph.

use super::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

type Results = BinaryHeap<Reverse<DistanceId>>;

impl HNSWIndex {
    /// Search for the `k` nearest live neighbours, skipping `deleted`.
    ///
    /// Deleted nodes are walked through but never returned. Returns
    /// `(id, distance)` pairs, closest first, or nothing for an empty index.
    ///
    /// # Errors
    /// Does not fail today: missing nodes are skipped. The `Result` is kept
    /// so a future check can report a corrupt index (rebuild it from
    /// vector storage) without an API change.
    pub fn search_filtered<F>(
        &self,
        query_id: u64,
        k: usize,
        distance_fn: &F,
        deleted: &HashSet<u64>,
    ) -> Result<Vec<(u64, f32)>>
    where
        F: Fn(u64, u64) -> f32,
    {
        self.search_accepted(query_id, k, distance_fn, &|id| !deleted.contains(&id))
    }

    /// Search for the `k` nearest neighbours for which `accept` is true.
    ///
    /// Nodes failing `accept` are used for navigation only. Returns
    /// `(id, distance)` pairs, closest first; fewer than `k` only when
    /// fewer than `k` accepted nodes are reachable from the entry point.
    /// Errors as [`HNSWIndex::search_filtered`].
    pub(crate) fn search_accepted<F, A>(
        &self,
        query_id: u64,
        k: usize,
        distance_fn: &F,
        accept: &A,
    ) -> Result<Vec<(u64, f32)>>
    where
        F: Fn(u64, u64) -> f32,
        A: Fn(u64) -> bool,
    {
        let Some(entry_id) = *self.entry_point.read() else {
            return Ok(Vec::new());
        };
        let top_level = *self.max_level.read();

        // Upper layers: plain greedy descent (rejected nodes are fine here)
        let mut curr_nearest = entry_id;
        for lc in (1..=top_level).rev() {
            curr_nearest = self.search_layer_single(query_id, curr_nearest, 1, lc, distance_fn)?;
        }

        let ef = self.config.ef_search.max(k);
        let mut found = self.layer0_accepted(query_id, curr_nearest, ef, distance_fn, accept);
        found.truncate(k);
        Ok(found)
    }

    fn layer0_accepted<F, A>(
        &self,
        query_id: u64,
        entry_id: u64,
        ef: usize,
        distance_fn: &F,
        accept: &A,
    ) -> Vec<(u64, f32)>
    where
        F: Fn(u64, u64) -> f32,
        A: Fn(u64) -> bool,
    {
        let nodes = self.nodes.read();
        // Two-hop expansion when under M of a node's (up to 2M) layer-0
        // links are accepted, i.e. when the filter rejects most of them.
        let two_hop_below = self.config.m.max(1);
        let mut visited = HashSet::from([entry_id]);
        // Nodes to expand, closest first (accepted and rejected)
        let mut candidates: BinaryHeap<DistanceId> = BinaryHeap::new();
        // Accepted results, farthest on top
        let mut results: Results = BinaryHeap::new();

        let bound = |results: &Results| {
            if results.len() < ef {
                f32::INFINITY
            } else {
                results
                    .peek()
                    .map_or(f32::INFINITY, |Reverse(d)| d.distance)
            }
        };
        // Score a node; queue it for expansion and keep it if accepted
        let consider = |id: u64, candidates: &mut BinaryHeap<DistanceId>, results: &mut Results| {
            let item = DistanceId {
                distance: distance_fn(query_id, id),
                id,
            };
            if item.distance >= bound(results) {
                return;
            }
            candidates.push(item);
            if accept(id) {
                results.push(Reverse(item));
                if results.len() > ef {
                    results.pop();
                }
            }
        };

        let entry = DistanceId {
            distance: distance_fn(query_id, entry_id),
            id: entry_id,
        };
        candidates.push(entry);
        if accept(entry_id) {
            results.push(Reverse(entry));
        }

        while let Some(curr) = candidates.pop() {
            if curr.distance > bound(&results) {
                break;
            }
            let Some(node) = nodes.get(&curr.id) else {
                continue;
            };
            let mut accepted = 0;
            let mut rejected = Vec::new();
            for &neighbor in &node.neighbors[0] {
                if accept(neighbor) {
                    accepted += 1;
                } else {
                    rejected.push(neighbor);
                }
                if visited.insert(neighbor) {
                    consider(neighbor, &mut candidates, &mut results);
                }
            }
            // Top the node up to M accepted neighbours from two hops away
            // (ACORN-1 truncation), so the expansion costs O(M^2) at most
            let mut budget = two_hop_below.saturating_sub(accepted);
            'hops: for hop in rejected {
                let Some(hop_node) = nodes.get(&hop) else {
                    continue;
                };
                for &second in &hop_node.neighbors[0] {
                    if budget == 0 {
                        break 'hops;
                    }
                    if accept(second) && visited.insert(second) {
                        budget -= 1;
                        consider(second, &mut candidates, &mut results);
                    }
                }
            }
        }

        let mut found: Vec<(u64, f32)> = results
            .into_iter()
            .map(|Reverse(d)| (d.id, d.distance))
            .collect();
        found.sort_by(|a, b| a.1.total_cmp(&b.1));
        found
    }
}
