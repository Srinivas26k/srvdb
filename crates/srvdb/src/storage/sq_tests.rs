//! Tests for SQ8 storage search

use super::*;
use tempfile::TempDir;

/// Deterministic values in [-1, 1) from a linear congruential generator
fn lcg_vectors(n: usize, dim: usize, mut seed: u64) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / (1u64 << 30) as f32) - 1.0
                })
                .collect()
        })
        .collect()
}

/// The scan is split into batches; ids, scores and tombstones must match a
/// one-by-one scoring of every stored code, across batch boundaries.
#[test]
fn search_matches_per_code_scoring_across_batches() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let dim = 37; // not a multiple of the SIMD lane count
    let data = lcg_vectors(1000, dim, 3);
    let mut storage = ScalarQuantizedStorage::new_with_training(path, dim, &data).unwrap();
    storage.append_batch(&data).unwrap();
    let deleted: HashSet<u64> = [0, 255, 256, 999].into_iter().collect();
    let query = &data[5];

    let mut want: Vec<(u64, f32)> = (0..data.len() as u64)
        .filter(|id| !deleted.contains(id))
        .map(|id| {
            let codes = storage.file.get(id).unwrap();
            (id, storage.quantizer.asymmetric_distance(query, codes))
        })
        .collect();
    want.sort_by(|a, b| b.1.total_cmp(&a.1));

    let got = storage.search(query, data.len(), &deleted).unwrap();
    assert_eq!(got.len(), want.len());
    let mut got_sorted = got.clone();
    got_sorted.sort_by_key(|r| r.0);
    let mut want_sorted = want.clone();
    want_sorted.sort_by_key(|r| r.0);
    for (g, w) in got_sorted.iter().zip(&want_sorted) {
        assert_eq!(g.0, w.0);
        assert!((g.1 - w.1).abs() <= 1e-5, "id {}: {} vs {}", g.0, g.1, w.1);
    }
    assert_eq!(got[0].0, 5, "the query itself scores highest");
}
