//! # Hybrid Search Engine
//!
//! Executes two-stage search:
//! 1. Coarse Search: Find best partitions using centroids.
//! 2. Fine Search: Search within those partitions using HNSW.

use crate::ivf::IVFIndex;
use crate::Vector;
use anyhow::Result;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub id: u64,
    pub score: f32, // Distance usually
}

// For min-heap to keep top-k
#[derive(Debug)]
struct ScoredItem {
    id: u64,
    score: f32,
}

impl PartialEq for ScoredItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for ScoredItem {}
impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by default in Rust
        // We want to keep K SMALLEST, so we use a Max-Heap to pop the LARGEST
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

pub fn search<F>(
    ivf: &IVFIndex,
    query: &Vector,
    k: usize,
    distance_fn: &F,
    deleted: &HashSet<u64>,
) -> Result<Vec<HybridSearchResult>>
where
    F: Fn(u64, u64) -> f32 + Sync,
{
    // 1. Coarse Search: Identify Closest Partitions
    // We want 'nprobe' partitions.
    let mut partition_scores: Vec<(usize, f32)> = ivf
        .centroids
        .iter()
        .enumerate()
        .map(|(i, center)| {
            // Centroid distance (Cosine Distance: 1.0 - Sim)
            let dist = 1.0 - crate::search::cosine_similarity(&query.data, &center.data);
            (i, dist)
        })
        .collect();

    // Sort by distance (asc)
    partition_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let target_partitions: Vec<usize> = partition_scores
        .iter()
        .take(ivf.config.nprobe)
        .map(|(i, _)| *i)
        .collect();

    if target_partitions.is_empty() {
        return Ok(Vec::new());
    }

    // 2. Fine Search (Parallel search in partitions)
    let candidates = fine_search(ivf, &target_partitions, k, distance_fn, deleted)?;

    // 3. Merge and formatting
    Ok(candidates)
}

/// Search within specific partitions and merge the top-k by distance
fn fine_search<F>(
    ivf: &IVFIndex,
    partitions: &[usize],
    k: usize,
    distance_fn: &F,
    deleted: &HashSet<u64>,
) -> Result<Vec<HybridSearchResult>>
where
    F: Fn(u64, u64) -> f32 + Sync,
{
    // Each partition is an independent HNSW graph; `search` starts from its
    // own entry point. The first argument is the query id placeholder.
    let all_candidates: Vec<Vec<(u64, f32)>> = partitions
        .par_iter()
        .filter(|&&part_id| part_id < ivf.partitions.len())
        .map(|&part_id| {
            ivf.partitions[part_id]
                .read()
                .search_filtered(0, k, distance_fn, deleted)
        })
        .collect::<Result<_>>()?;

    // Max-heap on distance keeps the k smallest
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (id, score) in all_candidates.into_iter().flatten() {
        heap.push(ScoredItem { id, score });
        if heap.len() > k {
            heap.pop();
        }
    }

    // Ascending distance: nearest first
    Ok(heap
        .into_sorted_vec()
        .into_iter()
        .map(|item| HybridSearchResult {
            id: item.id,
            score: item.score,
        })
        .collect())
}
