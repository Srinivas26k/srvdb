//! Product-quantization scan using asymmetric distance computation

use super::{select_top_k, BATCH_SIZE};
use crate::storage::pq::QuantizedVectorStorage;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashSet;

/// Optimized quantized search with Asymmetric Distance Computation
/// NOTE: Temporarily disabled - requires quantization.rs refactoring for dynamic dimensions
/// Scan of PQ codes using asymmetric distance, skipping `deleted` ids
pub fn search_quantized(
    storage: &QuantizedVectorStorage,
    query: &[f32],
    k: usize,
    deleted: &HashSet<u64>,
) -> Result<Vec<(u64, f32)>> {
    let count = storage.count() as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    // Distance table: query sub-vector to every centroid, computed once
    let dtable = storage.quantizer.compute_distance_table(query);
    let q_size = storage.q_size;

    let all_scores: Vec<(u64, f32)> = (0..count.div_ceil(BATCH_SIZE))
        .into_par_iter()
        .flat_map_iter(|batch| {
            let start = batch * BATCH_SIZE;
            let n = BATCH_SIZE.min(count - start);
            let codes = storage.get_batch(start as u64, n).unwrap_or(&[]);
            codes
                .chunks_exact(q_size)
                .enumerate()
                .map(move |(i, code)| ((start + i) as u64, code))
                .filter(|(id, _)| !deleted.contains(id))
                .map(|(id, code)| (id, storage.quantizer.asymmetric_distance(code, &dtable)))
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(select_top_k(all_scores, k))
}

/// Batch PQ search
pub fn search_quantized_batch(
    storage: &QuantizedVectorStorage,
    queries: &[Vec<f32>],
    k: usize,
    deleted: &HashSet<u64>,
) -> Result<Vec<Vec<(u64, f32)>>> {
    queries
        .par_iter()
        .map(|query| search_quantized(storage, query, k, deleted))
        .collect()
}
