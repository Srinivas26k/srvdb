//! Neighbour selection and bidirectional link maintenance

use super::*;
use std::cmp::Ordering;

impl HNSWIndex {
    /// Diversity heuristic (Algorithm 4 - SELECT-NEIGHBORS-HEURISTIC)
    ///
    /// `candidates` must be sorted nearest first. A candidate is kept only
    /// if it is closer to the base node than to every neighbour already
    /// kept. Keeping just the M nearest instead fills every list with
    /// same-cluster nodes and drops the long links between clusters, which
    /// leaves clustered data as disconnected islands (issue #28).
    pub(super) fn select_neighbors_heuristic<F>(
        &self,
        candidates: &[(u64, f32)],
        m: usize,
        distance_fn: &F,
    ) -> Vec<u64>
    where
        F: Fn(u64, u64) -> f32,
    {
        let mut kept: Vec<u64> = Vec::with_capacity(m);
        for &(cand, dist_to_base) in candidates {
            if kept.len() >= m {
                break;
            }
            if kept.iter().all(|&r| dist_to_base < distance_fn(cand, r)) {
                kept.push(cand);
            }
        }
        kept
    }

    /// Add a directed link and prune with the heuristic if the list is full
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
            if !from_node.neighbors[layer].contains(&to_id) {
                from_node.neighbors[layer].push(to_id);

                let m_max = if layer == 0 {
                    self.config.m_max0
                } else {
                    self.config.m_max
                };

                if from_node.neighbors[layer].len() > m_max {
                    let mut neighbor_dists: Vec<_> = from_node.neighbors[layer]
                        .iter()
                        .map(|&nid| (nid, distance_fn(from_id, nid)))
                        .collect();
                    neighbor_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    from_node.neighbors[layer] =
                        self.select_neighbors_heuristic(&neighbor_dists, m_max, distance_fn);
                }
            }
        }

        Ok(())
    }
}
