//! Filtered search (issue #25): recall against exact filtered search, and
//! always k results when at least k vectors match.

use srvdb::{HNSWConfig, SrvDB, Vector, VectorEngine};
use std::collections::HashSet;
use tempfile::TempDir;

const DIM: usize = 32;
const N: usize = 3000;
const K: usize = 10;

/// SplitMix64 + Box-Muller: deterministic, independent of `rand` versions
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// Seeded Gaussian clusters plus a `bucket` in 0..100 per vector,
/// independent of the cluster (so a filter cuts across clusters)
fn clustered(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<u64>) {
    let mut rng = SplitMix(seed);
    let centres: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..DIM).map(|_| rng.gauss()).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centres[(rng.next_u64() % 20) as usize];
            let v = c.iter().map(|x| x + 0.6 * rng.gauss()).collect();
            (v, rng.next_u64() % 100)
        })
        .unzip()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn hnsw_db(dir: &TempDir, base: &[Vec<f32>], buckets: &[u64]) -> SrvDB {
    let path = dir.path().to_str().unwrap();
    let mut db = SrvDB::new_with_hnsw(path, DIM, HNSWConfig::new(16, 100, 64)).unwrap();
    let vecs: Vec<Vector> = base.iter().cloned().map(Vector::new).collect();
    let metas: Vec<String> = buckets
        .iter()
        .map(|b| format!("{{\"bucket\":{b}}}"))
        .collect();
    db.add_batch(&vecs, &metas).unwrap();
    db
}

/// Ids whose bucket is below `percent`, via the metadata predicate API
fn allowed(db: &SrvDB, percent: u64) -> HashSet<u64> {
    db.ids_where(|meta| {
        let v: serde_json::Value = serde_json::from_str(meta).unwrap();
        v["bucket"].as_u64().unwrap() < percent
    })
    .unwrap()
}

fn filtered_recall(percent: u64) -> f64 {
    let (all, buckets) = clustered(N + 50, 25);
    let (base, queries) = all.split_at(N);
    let dir = TempDir::new().unwrap();
    let db = hnsw_db(&dir, base, &buckets[..N]);
    let allow = allowed(&db, percent);
    assert!(allow.len() >= K, "test set needs at least k matches");

    let mut hits = 0;
    for q in queries {
        let mut exact: Vec<(u64, f32)> = allow
            .iter()
            .map(|&id| (id, cosine(q, &base[id as usize])))
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));
        let truth: HashSet<u64> = exact[..K].iter().map(|&(id, _)| id).collect();
        let found = db
            .search_filtered(&Vector::new(q.clone()), K, &allow)
            .unwrap();
        assert_eq!(found.len(), K, "at {percent}% fewer than k results");
        assert!(found.iter().all(|r| allow.contains(&r.id)));
        hits += found.iter().filter(|r| truth.contains(&r.id)).count();
    }
    hits as f64 / (queries.len() * K) as f64
}

#[test]
fn recall_at_10_with_1_percent_matching() {
    let r = filtered_recall(1);
    assert!(r >= 0.95, "filtered recall@10 at 1%: {r:.3}");
}

#[test]
fn recall_at_10_with_10_percent_matching() {
    let r = filtered_recall(10);
    assert!(r >= 0.95, "filtered recall@10 at 10%: {r:.3}");
}

#[test]
fn recall_at_10_with_50_percent_matching() {
    let r = filtered_recall(50);
    assert!(r >= 0.95, "filtered recall@10 at 50%: {r:.3}");
}

/// Filtering after search (the only option before #25) loses results;
/// filtering during search returns k whenever k vectors match.
#[test]
fn filtered_search_returns_k_where_post_filtering_does_not() {
    let (base, buckets) = clustered(N, 26);
    let dir = TempDir::new().unwrap();
    let db = hnsw_db(&dir, &base, &buckets);
    let allow = allowed(&db, 10);
    let query = Vector::new(base[0].clone());

    let post: Vec<u64> = db
        .search(&query, K)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .filter(|id| allow.contains(id))
        .collect();
    assert!(post.len() < K, "post-filter kept {} of {K}", post.len());

    let found = db.search_filtered(&query, K, &allow).unwrap();
    assert_eq!(found.len(), K);
    assert!(found.iter().all(|r| r.metadata.is_some()));
}

#[test]
fn deleted_and_unknown_ids_are_never_returned() {
    let (base, buckets) = clustered(500, 27);
    let dir = TempDir::new().unwrap();
    let mut db = hnsw_db(&dir, &base, &buckets);
    let mut allow: HashSet<u64> = (0..20).collect();
    allow.insert(1_000_000);
    for id in 0..5 {
        db.delete(id).unwrap();
    }
    let found = db
        .search_filtered(&Vector::new(base[0].clone()), 50, &allow)
        .unwrap();
    let ids: HashSet<u64> = found.iter().map(|r| r.id).collect();
    assert_eq!(ids, (5..20).collect());
}

#[test]
fn flat_mode_scores_the_allowed_ids_exactly() {
    // Flat databases need at least 128 dimensions; zero padding keeps cosine
    let (mut base, buckets) = clustered(500, 28);
    base.iter_mut().for_each(|v| v.resize(128, 0.0));
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let mut db = SrvDB::new_with_config(path, srvdb::DatabaseConfig::new(128).unwrap()).unwrap();
    let vecs: Vec<Vector> = base.iter().cloned().map(Vector::new).collect();
    let metas: Vec<String> = buckets
        .iter()
        .map(|b| format!("{{\"bucket\":{b}}}"))
        .collect();
    db.add_batch(&vecs, &metas).unwrap();
    let allow = allowed(&db, 10);
    let found = db
        .search_filtered(&Vector::new(base[1].clone()), K, &allow)
        .unwrap();
    assert_eq!(found.len(), K);
    let best = allow
        .iter()
        .max_by(|a, b| {
            cosine(&base[1], &base[**a as usize]).total_cmp(&cosine(&base[1], &base[**b as usize]))
        })
        .unwrap();
    assert_eq!(found[0].id, *best);
}

#[test]
fn zero_k_is_rejected() {
    let (base, buckets) = clustered(50, 29);
    let dir = TempDir::new().unwrap();
    let db = hnsw_db(&dir, &base, &buckets);
    let allow: HashSet<u64> = (0..10).collect();
    assert!(db
        .search_filtered(&Vector::new(base[0].clone()), 0, &allow)
        .is_err());
}
