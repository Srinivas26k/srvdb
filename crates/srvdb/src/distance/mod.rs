//! Ultra-fast SIMD-accelerated k-NN search
//!
//! Optimizations:
//! - Batch processing with 256-vector chunks
//! - SIMD intrinsics for cosine similarity
//! - Lock-free parallel heap for top-k selection
//! - Cache-optimized memory access patterns

mod graph;
mod quantized;

pub use graph::{search_hnsw, search_hnsw_quantized};
pub use quantized::{search_quantized, search_quantized_batch};

use crate::VectorStorage;
use anyhow::Result;
use rayon::prelude::*;
use simsimd::SpatialSimilarity;
use std::cmp::Ordering;
use std::collections::HashSet;

pub(crate) const BATCH_SIZE: usize = 256; // Process 256 vectors at once for cache efficiency

#[inline]
/// Cosine similarity in [-1, 1] (1 = same direction). simsimd returns the
/// cosine distance, 1 - cos; a failed computation counts as opposite (-1).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let distance = f32::cosine(a, b).unwrap_or(2.0) as f32;
    1.0 - distance
}

#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    f32::sqeuclidean(a, b).unwrap_or(0.0) as f32
}

/// Batch compute similarities for cache efficiency
#[inline]
fn compute_similarities_batch(
    query: &[f32],
    vectors: &[&[f32]],
    start_id: u64,
    deleted: &HashSet<u64>,
) -> Vec<(u64, f32)> {
    vectors
        .iter()
        .enumerate()
        .filter(|(i, _)| !deleted.contains(&(start_id + *i as u64)))
        .map(|(i, vec)| {
            let score = cosine_similarity(query, vec);
            (start_id + i as u64, score)
        })
        .collect()
}

/// Fast top-k selection using partial sorting
pub(crate) fn select_top_k(mut candidates: Vec<(u64, f32)>, k: usize) -> Vec<(u64, f32)> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let k = k.min(candidates.len());

    // Partial sort: only sort enough to get top-k
    candidates.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
    });

    // Take top-k and sort them
    let mut top_k: Vec<_> = candidates.into_iter().take(k).collect();
    top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    top_k
}

/// Optimized parallel search with batch processing
/// Exact search over full-precision vectors, skipping `deleted` ids
pub fn search_cosine(
    storage: &VectorStorage,
    query: &[f32],
    k: usize,
    deleted: &HashSet<u64>,
) -> Result<Vec<(u64, f32)>> {
    let count = storage.count() as usize;

    if count == 0 {
        return Ok(Vec::new());
    }

    let actual_k = k.min(count);

    // Process in cache-friendly batches
    let num_batches = count.div_ceil(BATCH_SIZE);

    let all_similarities: Vec<(u64, f32)> = (0..num_batches)
        .into_par_iter()
        .flat_map(|batch_idx| {
            let start = batch_idx * BATCH_SIZE;
            let batch_size = BATCH_SIZE.min(count - start);

            // Get batch of vectors (zero-copy)
            if let Some(vectors) = storage.get_batch(start as u64, batch_size) {
                compute_similarities_batch(query, &vectors, start as u64, deleted)
            } else {
                Vec::new()
            }
        })
        .collect();

    // Fast top-k selection
    Ok(select_top_k(all_similarities, actual_k))
}

/// Hyper-optimized multi-query search (for concurrent throughput)
pub fn search_batch(
    storage: &VectorStorage,
    queries: &[Vec<f32>],
    k: usize,
    deleted: &HashSet<u64>,
) -> Result<Vec<Vec<(u64, f32)>>> {
    queries
        .par_iter()
        .map(|query| search_cosine(storage, query.as_slice(), k, deleted))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_is_the_true_cosine() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let close = [1.0f32, 1.0, 0.0, 0.0];
        let orthogonal = [0.0f32, 1.0, 0.0, 0.0];
        let opposite = [-2.0f32, 0.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
        assert!((cosine_similarity(&a, &close) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        assert!(cosine_similarity(&a, &orthogonal).abs() < 1e-5);
        assert!((cosine_similarity(&a, &opposite) + 1.0).abs() < 1e-5);
    }
    use crate::VectorStorage;
    use tempfile::TempDir;

    #[test]
    fn test_batch_search() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(temp_dir.path().to_str().unwrap(), 1536).unwrap();

        // Add test vectors
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| {
                let mut v = vec![0.0f32; 1536];
                v[0] = (i as f32) / 1000.0;
                v
            })
            .collect();

        // Convert to slice of slices for append_batch
        // Note: append_batch expects &[Vec<f32>]
        storage.append_batch(&vectors).unwrap();
        storage.flush().unwrap();

        // Create multiple queries
        let queries: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut v = vec![0.0f32; 1536];
                v[0] = (i as f32) / 10.0;
                v
            })
            .collect();

        // Convert queries to required format
        let query_slices: Vec<Vec<f32>> = queries.clone();

        let results = search_batch(&storage, &query_slices, 5, &HashSet::new()).unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.len() == 5));
    }

    #[test]
    fn test_partial_sort_performance() {
        let candidates: Vec<(u64, f32)> = (0..10000).map(|i| (i, rand::random::<f32>())).collect();

        let top_k = select_top_k(candidates, 10);
        assert_eq!(top_k.len(), 10);

        // Verify descending order
        for i in 1..top_k.len() {
            assert!(top_k[i - 1].1 >= top_k[i].1);
        }
    }
}
