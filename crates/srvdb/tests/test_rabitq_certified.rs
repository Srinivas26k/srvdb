//! Certified and deadline search (ADR 0017): measured recall against
//! exact search meets the claimed confidence on many seeded datasets;
//! deadline 0 is partial; a generous deadline equals normal search.

use srvdb::{
    DatabaseConfig, RabitqConfig, RabitqSearch, SearchResult, SrvDB, Vector, VectorEngine,
};
use std::collections::HashSet;
use std::time::Duration;
use tempfile::TempDir;

const DIM: usize = 96;
const K: usize = 10;

fn rng(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

/// Clustered data whose cluster count and spread vary with the seed
fn clustered(n: usize, queries: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let clusters = 5 + (seed as usize % 40);
    let spread = 0.1 + 0.1 * (seed % 5) as f32;
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..DIM).map(|_| rng(&mut s)).collect())
        .collect();
    let mut draw = |i: usize| -> Vec<f32> {
        centres[i % clusters]
            .iter()
            .map(|x| x + spread * rng(&mut s))
            .collect()
    };
    let base = (0..n).map(&mut draw).collect();
    let qs = (n..n + queries).map(&mut draw).collect();
    (base, qs)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (n(a) * n(b))
}

fn exact(base: &[Vec<f32>], q: &[f32]) -> HashSet<u64> {
    let mut all: Vec<(u64, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, cosine(q, v)))
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    all.into_iter().take(K).map(|(i, _)| i).collect()
}

fn build(dir: &TempDir, bits: u8, seed: u64, base: &[Vec<f32>]) -> SrvDB {
    let config = RabitqConfig::new(bits).unwrap().with_seed(seed);
    let mut db = SrvDB::new_rabitq(dir.path().to_str().unwrap(), DIM, config).unwrap();
    let vecs: Vec<Vector> = base.iter().cloned().map(Vector::new).collect();
    let metas = vec!["{}".to_string(); vecs.len()];
    db.add_batch(&vecs, &metas).unwrap();
    db
}

#[test]
fn certified_recall_meets_confidence_on_seeded_datasets() {
    let options = RabitqSearch::new().with_certify(true);
    for seed in 1..=8u64 {
        for bits in [1u8, 2, 4] {
            let (base, queries) = clustered(1500, 20, seed * 31 + bits as u64);
            let dir = TempDir::new().unwrap();
            let db = build(&dir, bits, seed, &base);
            let (mut hits, mut certified, mut reranked) = (0usize, 0, 0);
            let mut confidence = 1.0;
            for q in &queries {
                let query = Vector::new(q.clone());
                let out = db.search_rabitq_with(&query, K, &options).unwrap();
                assert_eq!(out.results.len(), K);
                assert_eq!(out.stats.completeness, 1.0);
                confidence = out.stats.confidence.unwrap();
                certified += out.stats.certified;
                reranked += out.stats.reranked;
                let truth = exact(&base, q);
                hits += out.results.iter().filter(|r| truth.contains(&r.id)).count();
            }
            let recall = hits as f64 / (K * queries.len()) as f64;
            assert!(
                recall >= confidence,
                "seed {seed} bits {bits}: recall {recall} < confidence {confidence} \
                 (certified {certified}, reranked {reranked})"
            );
        }
    }
}

fn pairs(r: &[SearchResult]) -> Vec<(u64, f32)> {
    r.iter().map(|x| (x.id, x.score)).collect()
}

#[test]
fn deadline_zero_is_partial_and_generous_deadline_matches_search() {
    let (base, queries) = clustered(3000, 5, 99);
    let dir = TempDir::new().unwrap();
    let db = build(&dir, 4, 5, &base);
    for q in &queries {
        let query = Vector::new(q.clone());
        for certify in [false, true] {
            let zero = RabitqSearch::new()
                .with_certify(certify)
                .with_deadline(Duration::ZERO);
            let out = db.search_rabitq_with(&query, K, &zero).unwrap();
            assert!(out.stats.completeness < 1.0);
        }
        let generous = RabitqSearch::new().with_deadline(Duration::from_secs(3600));
        let out = db.search_rabitq_with(&query, K, &generous).unwrap();
        assert_eq!(out.stats.completeness, 1.0);
        assert_eq!(pairs(&out.results), pairs(&db.search(&query, K).unwrap()));
    }
}

#[test]
fn other_modes_are_refused() {
    let dir = TempDir::new().unwrap();
    let config = DatabaseConfig::new(128).unwrap();
    let db = SrvDB::new_with_config(dir.path().to_str().unwrap(), config).unwrap();
    let options = RabitqSearch::new().with_certify(true);
    let err = db
        .search_rabitq_with(&Vector::new(vec![0.5; 128]), K, &options)
        .unwrap_err();
    assert!(matches!(err, srvdb::Error::InvalidInput(_)), "{err}");
}

/// Well-separated neighbours are certified without rescoring: each query
/// is a stored vector among unrelated random ones
#[test]
fn separated_neighbours_are_certified() {
    let mut s = 77u64;
    let base: Vec<Vec<f32>> = (0..2000)
        .map(|_| (0..DIM).map(|_| rng(&mut s)).collect())
        .collect();
    let dir = TempDir::new().unwrap();
    let db = build(&dir, 4, 3, &base);
    let options = RabitqSearch::new().with_certify(true);
    for i in [0usize, 500, 1999] {
        let out = db
            .search_rabitq_with(&Vector::new(base[i].clone()), 1, &options)
            .unwrap();
        assert_eq!(out.results[0].id, i as u64);
        assert_eq!(out.stats.certified, 1);
        assert_eq!(out.stats.reranked, 0);
    }
}
