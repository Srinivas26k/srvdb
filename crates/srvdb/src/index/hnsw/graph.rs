//! Graph construction and traversal (Algorithms 1, 2, 4 and 5 of the paper)

use super::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

impl HNSWIndex {
    /// Insert a new vector into the HNSW graph
    ///
    /// This is the core algorithm from the paper (Algorithm 1)
    pub fn insert<F>(&self, id: u64, distance_fn: &F) -> Result<()>
    where
        F: Fn(u64, u64) -> f32,
    {
        let level = self.random_level();
        let node = HNSWNode::new(id, level);

        // Lock order everywhere: entry_point, then max_level, then nodes.
        // Checking for an empty graph and claiming the entry point is one
        // atomic step: otherwise concurrent first inserts would each add an
        // unlinked node, and all but one would be unreachable forever.
        // The entry point and top level are read together, under the lock,
        // so the search never starts above the entry node's own level.
        let (entry_id, top_level) = {
            let mut entry_point = self.entry_point.write();
            match *entry_point {
                Some(entry_id) => (entry_id, *self.max_level.read()),
                None => {
                    *self.max_level.write() = level;
                    self.nodes.write().insert(id, node);
                    *entry_point = Some(id);
                    return Ok(());
                }
            }
        };

        // Add the node (with empty neighbour lists) before linking, so links
        // other threads make to it while it is being connected are kept
        // instead of silently dropped.
        self.nodes.write().insert(id, node);

        {
            // Graph already has nodes
            let mut curr_nearest = entry_id;

            // Phase 1: Greedy search from top to target layer. The node being
            // inserted is already in the graph and may be reachable through
            // links other inserts made to it; never descend into it, since
            // its own lists are still empty.
            for lc in (level + 1..=top_level).rev() {
                let nearest = self.search_layer(id, curr_nearest, 2, lc, distance_fn)?;
                if let Some(&(next, _)) = nearest.iter().find(|&&(c, _)| c != id) {
                    curr_nearest = next;
                }
            }

            // Phase 2: Search and connect at each layer from level down to 0
            for lc in (0..=level).rev() {
                // If this layer is higher than the current max level, there are no existing nodes here
                if lc > top_level {
                    continue;
                }

                let candidates = self.search_layer(
                    id,
                    curr_nearest,
                    self.config.ef_construction,
                    lc,
                    distance_fn,
                )?;

                // Get M nearest neighbors
                let m = if lc == 0 {
                    self.config.m_max0
                } else {
                    self.config.m_max
                };

                // A concurrent insert may already have linked to this node
                let candidates: Vec<(u64, f32)> =
                    candidates.into_iter().filter(|&(c, _)| c != id).collect();
                if candidates.is_empty() {
                    continue;
                }
                let neighbors = self.select_neighbors_simple(&candidates, m);

                // Link both ways, merging with links added concurrently
                for &neighbor_id in &neighbors {
                    self.add_bidirectional_link(id, neighbor_id, lc, distance_fn)?;
                    self.add_bidirectional_link(neighbor_id, id, lc, distance_fn)?;
                }

                curr_nearest = neighbors[0];
            }
        }

        // Re-check under the lock: another insert may have raised the top
        // level since `top_level` was read.
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();
        if level > *max_level {
            *max_level = level;
            *entry_point = Some(id);
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// This is Algorithm 5 from the paper
    pub fn search<F>(&self, query_id: u64, k: usize, distance_fn: &F) -> Result<Vec<(u64, f32)>>
    where
        F: Fn(u64, u64) -> f32,
    {
        let Some(entry_id) = *self.entry_point.read() else {
            return Ok(Vec::new());
        };
        let mut curr_nearest = entry_id;
        let top_level = *self.max_level.read();

        // Phase 1: Greedy search from top to layer 1
        for lc in (1..=top_level).rev() {
            curr_nearest = self.search_layer_single(query_id, curr_nearest, 1, lc, distance_fn)?;
        }

        // Phase 2: Search at layer 0 with ef_search
        let candidates = self.search_layer(
            query_id,
            curr_nearest,
            self.config.ef_search.max(k),
            0,
            distance_fn,
        )?;

        // Return top-k with distances
        Ok(candidates.into_iter().take(k).collect())
    }

    /// Search a single layer (Algorithm 2 from paper)
    /// Returns the single nearest neighbor
    pub(super) fn search_layer_single<F>(
        &self,
        query_id: u64,
        entry_id: u64,
        num_closest: usize,
        layer: usize,
        distance_fn: &F,
    ) -> Result<u64>
    where
        F: Fn(u64, u64) -> f32,
    {
        let results = self.search_layer(query_id, entry_id, num_closest, layer, distance_fn)?;
        Ok(results[0].0)
    }

    /// Search a single layer (Algorithm 2 from paper)
    /// Returns ef nearest neighbors
    fn search_layer<F>(
        &self,
        query_id: u64,
        entry_id: u64,
        ef: usize,
        layer: usize,
        distance_fn: &F,
    ) -> Result<Vec<(u64, f32)>>
    where
        F: Fn(u64, u64) -> f32,
    {
        let nodes = self.nodes.read();
        let mut visited = HashSet::new();

        // Candidates: min-heap (closest first)
        let mut candidates: BinaryHeap<DistanceId> = BinaryHeap::new();

        // Results: max-heap by distance (farthest on top for easy pruning)
        let mut results: BinaryHeap<Reverse<DistanceId>> = BinaryHeap::new();

        let entry_dist = distance_fn(query_id, entry_id);
        let entry_item = DistanceId {
            distance: entry_dist,
            id: entry_id,
        };

        candidates.push(entry_item);
        results.push(Reverse(entry_item));
        visited.insert(entry_id);

        while let Some(curr) = candidates.pop() {
            // If current is farther than the farthest in results, stop
            if let Some(&Reverse(farthest)) = results.peek() {
                if curr.distance > farthest.distance {
                    break;
                }
            }

            // Explore neighbors
            if let Some(node) = nodes.get(&curr.id) {
                if layer < node.neighbors.len() {
                    for &neighbor_id in &node.neighbors[layer] {
                        if !visited.insert(neighbor_id) {
                            continue; // Already visited
                        }

                        let neighbor_dist = distance_fn(query_id, neighbor_id);

                        // Check if this neighbor should be considered
                        let should_add = if results.len() < ef {
                            true
                        } else if let Some(&Reverse(farthest)) = results.peek() {
                            neighbor_dist < farthest.distance
                        } else {
                            false
                        };

                        if should_add {
                            let neighbor_item = DistanceId {
                                distance: neighbor_dist,
                                id: neighbor_id,
                            };
                            candidates.push(neighbor_item);
                            results.push(Reverse(neighbor_item));

                            if results.len() > ef {
                                results.pop(); // Remove farthest
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vector (nearest first)
        let mut result_vec: Vec<_> = results
            .into_iter()
            .map(|Reverse(di)| (di.id, di.distance))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(result_vec)
    }
}
