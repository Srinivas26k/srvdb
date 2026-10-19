//! Product-quantization scan using asymmetric distance computation

use super::{select_top_k, BATCH_SIZE};
use crate::storage::pq::QuantizedVectorStorage;
use anyhow::Result;
use rayon::prelude::*;

/// Optimized quantized search with Asymmetric Distance Computation
/// NOTE: Temporarily disabled - requires quantization.rs refactoring for dynamic dimensions
pub fn search_quantized(
    storage: &QuantizedVectorStorage,
    query: &[f32],
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    let count = storage.count();
    if count == 0 {
        return Ok(Vec::new());
    }

    // 1. Precompute Distance Table (O(D * K))
    // We assume the query is already normalized or handled by quantizer
    let dtable = storage.quantizer.compute_distance_table(query);

    // 2. Scan all vectors (O(N * M))
    // Use parallel iterator for speed
    let num_batches = (count as usize).div_ceil(BATCH_SIZE);

    let all_scores: Vec<(u64, f32)> = (0..num_batches)
        .into_par_iter()
        .flat_map(|batch_idx| {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(count as usize);
            let mut results = Vec::with_capacity(end - start);

            // Access local Mmap slice for this batch if possible? 
            // storage.get() does bounds check every time. 
            // get_batch() is better.
            if let Some(batch_data) = storage.get_batch(start as u64, end - start) {
                // batch_data is a flat slice of [u8].
                // We know each vector has storage.q_size bytes.
                let q_size = storage.q_size;
                
                for (i, chunk) in batch_data.chunks(q_size).enumerate() {
                    if chunk.len() == q_size {
                         let dist = storage.quantizer.asymmetric_distance(chunk, &dtable);
                         results.push(((start + i) as u64, dist));
                    }
                }
            } else {
                 // Fallback to individual gets if batch fails (shouldn't happen)
                for i in start..end {
                    if let Some(qvec) = storage.get(i as u64) {
                        let dist = storage.quantizer.asymmetric_distance(qvec, &dtable);
                        results.push((i as u64, dist));
                    }
                }
            }
            results
        })
        .collect();

    Ok(select_top_k(all_scores, k))
}

/// Batch quantized search
/// NOTE: Temporarily disabled
pub fn search_quantized_batch(
    storage: &QuantizedVectorStorage,
    queries: &[Vec<f32>],
    k: usize,
) -> Result<Vec<Vec<(u64, f32)>>> {
    queries
        .par_iter()
        .map(|query| search_quantized(storage, query, k))
        .collect()
}
