//! Graph (HNSW) search over full-precision and PQ storage

use super::cosine_similarity;
use crate::storage::pq::QuantizedVectorStorage;
use crate::VectorStorage;
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::HashSet;

/// HNSW search for full-precision vectors
///
/// Uses graph-based approximate nearest neighbor search for O(log n) complexity
pub fn search_hnsw(
    storage: &VectorStorage,
    hnsw: &crate::hnsw::HNSWIndex,
    query: &[f32],
    k: usize,
    deleted: &HashSet<u64>,
) -> Result<Vec<(u64, f32)>> {
    // HNSW graph traversal with query-to-storage distance function
    let candidates = hnsw.search_filtered(
        0,
        k,
        &|_query_id: u64, vec_id: u64| -> f32 {
            if let Some(vec) = storage.get(vec_id) {
                1.0 - cosine_similarity(query, vec) // Distance (lower is better)
            } else {
                f32::MAX
            }
        },
        deleted,
    )?;

    // Convert back to similarities (higher is better) and take top-k
    let mut results: Vec<(u64, f32)> = candidates
        .into_iter()
        .map(|(id, dist)| (id, 1.0 - dist)) // Convert distance back to similarity
        .collect();

    // Sort by similarity (descending)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    results.truncate(k);

    Ok(results)
}

/// HNSW search for quantized vectors
/// NOTE: Temporarily disabled
pub fn search_hnsw_quantized(
    storage: &QuantizedVectorStorage,
    hnsw: &crate::hnsw::HNSWIndex,
    query: &[f32],
    k: usize,
    deleted: &HashSet<u64>,
) -> Result<Vec<(u64, f32)>> {
    // Precompute lookup table for this query
    let dtable = storage.quantizer.compute_distance_table(query);

    // HNSW graph traversal using ADC
    let candidates = hnsw.search_filtered(
        0,
        k,
        &|_query_id: u64, vec_id: u64| -> f32 {
            if let Some(qvec) = storage.get(vec_id) {
                let sim = storage.quantizer.asymmetric_distance(qvec, &dtable);
                // Convert Similarity to Distance (HNSW minimizes)
                // Sim is typically 0.0 to 1.0 (approx cosine)
                1.0 - sim
            } else {
                f32::MAX
            }
        },
        deleted,
    )?;

    // Convert back to similarities (higher is better) and take top-k
    let mut results: Vec<(u64, f32)> = candidates
        .into_iter()
        .map(|(id, dist)| (id, 1.0 - dist)) // Convert distance back to similarity
        .collect();

    // Sort by similarity (descending)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    results.truncate(k);

    Ok(results)
}
