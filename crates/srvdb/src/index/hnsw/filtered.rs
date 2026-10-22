//! Search that skips deleted ids (ADR 0004)
//!
//! Deleted nodes stay in the graph: they keep it connected, so the search
//! still walks through them, but they are never returned. The layer-0
//! search keeps exploring until it holds `ef` live results, so a search
//! returns `k` results whenever `k` live nodes are reachable.

use super::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

impl HNSWIndex {
    /// Search for the `k` nearest live neighbours, skipping `deleted`
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
        let Some(entry_id) = *self.entry_point.read() else {
            return Ok(Vec::new());
        };
        let top_level = *self.max_level.read();

        // Upper layers: plain greedy descent (deleted nodes are fine here)
        let mut curr_nearest = entry_id;
        for lc in (1..=top_level).rev() {
            curr_nearest = self.search_layer_single(query_id, curr_nearest, 1, lc, distance_fn)?;
        }

        let ef = self.config.ef_search.max(k);
        let mut found = self.search_layer0_live(query_id, curr_nearest, ef, distance_fn, deleted);
        found.truncate(k);
        Ok(found)
    }

    fn search_layer0_live<F>(
        &self,
        query_id: u64,
        entry_id: u64,
        ef: usize,
        distance_fn: &F,
        deleted: &HashSet<u64>,
    ) -> Vec<(u64, f32)>
    where
        F: Fn(u64, u64) -> f32,
    {
        let nodes = self.nodes.read();
        let mut visited = HashSet::from([entry_id]);
        // Nodes to expand, closest first (live and deleted)
        let mut candidates: BinaryHeap<DistanceId> = BinaryHeap::new();
        // Live results, farthest on top
        let mut live: BinaryHeap<Reverse<DistanceId>> = BinaryHeap::new();

        let entry = DistanceId {
            distance: distance_fn(query_id, entry_id),
            id: entry_id,
        };
        candidates.push(entry);
        if !deleted.contains(&entry_id) {
            live.push(Reverse(entry));
        }

        let bound = |live: &BinaryHeap<Reverse<DistanceId>>| {
            if live.len() < ef {
                f32::INFINITY
            } else {
                live.peek().map_or(f32::INFINITY, |Reverse(d)| d.distance)
            }
        };

        while let Some(curr) = candidates.pop() {
            if curr.distance > bound(&live) {
                break;
            }
            let Some(node) = nodes.get(&curr.id) else {
                continue;
            };
            for &neighbor in &node.neighbors[0] {
                if !visited.insert(neighbor) {
                    continue;
                }
                let item = DistanceId {
                    distance: distance_fn(query_id, neighbor),
                    id: neighbor,
                };
                if item.distance >= bound(&live) {
                    continue;
                }
                candidates.push(item);
                if !deleted.contains(&neighbor) {
                    live.push(Reverse(item));
                    if live.len() > ef {
                        live.pop();
                    }
                }
            }
        }

        let mut found: Vec<(u64, f32)> = live
            .into_iter()
            .map(|Reverse(d)| (d.id, d.distance))
            .collect();
        found.sort_by(|a, b| a.1.total_cmp(&b.1));
        found
    }
}
