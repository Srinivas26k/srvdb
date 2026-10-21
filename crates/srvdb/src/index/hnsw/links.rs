//! Neighbour selection and bidirectional link maintenance

use super::*;
use std::cmp::Ordering;

impl HNSWIndex {
    /// Simple heuristic for neighbor selection (Algorithm 3 - SELECT-NEIGHBORS-SIMPLE)
    /// Just returns the M nearest neighbors
    pub(super) fn select_neighbors_simple(&self, candidates: &[(u64, f32)], m: usize) -> Vec<u64> {
        candidates.iter().take(m).map(|(id, _)| *id).collect()
    }

    /// Add a bidirectional link and prune if necessary (Algorithm 4)
    pub(super) fn add_bidirectional_link<F>(
        &self,
        from_id: u64,
        to_id: u64,
        layer: usize,
        distance_fn: &F,
    ) -> Result<()>
    where
        F: Fn(u64, u64) -> f32,
    {
        let mut nodes = self.nodes.write();

        // Skip nodes that do not exist (yet) or do not reach this layer
        if let Some(from_node) = nodes
            .get_mut(&from_id)
            .filter(|node| layer < node.neighbors.len())
        {
            // Add link if not already present
            if !from_node.neighbors[layer].contains(&to_id) {
                from_node.neighbors[layer].push(to_id);

                // Prune if exceeds max connections
                let m_max = if layer == 0 {
                    self.config.m_max0
                } else {
                    self.config.m_max
                };

                if from_node.neighbors[layer].len() > m_max {
                    // Calculate distances to all neighbors
                    let mut neighbor_dists: Vec<_> = from_node.neighbors[layer]
                        .iter()
                        .map(|&nid| {
                            let dist = distance_fn(from_id, nid);
                            (nid, dist)
                        })
                        .collect();

                    // Sort by distance and keep only M nearest
                    neighbor_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    from_node.neighbors[layer] = neighbor_dists
                        .into_iter()
                        .take(m_max)
                        .map(|(nid, _)| nid)
                        .collect();
                }
            }
        }

        Ok(())
    }
}
