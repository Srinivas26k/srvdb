//! `SrvDB::search_where` (issue #46): a declarative metadata filter gives
//! the same results as the id-set API, returns k results when k match, and
//! rejects bad filters as invalid input.

use srvdb::{Error, Filter, HNSWConfig, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 16;
const N: u64 = 1200;

/// Deterministic pseudo-random values in [-1, 1)
fn values(seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn hnsw_db(dir: &TempDir) -> SrvDB {
    let path = dir.path().to_str().unwrap();
    let mut db = SrvDB::new_with_hnsw(path, DIM, HNSWConfig::new(16, 100, 64)).unwrap();
    let vecs: Vec<Vector> = (0..N).map(|i| Vector::new(values(i + 1))).collect();
    let metas: Vec<String> = (0..N)
        .map(|i| {
            format!(
                r#"{{"meta":{{"bucket":{},"lang":"{}"}}}}"#,
                i % 100,
                ["en", "de"][(i % 2) as usize]
            )
        })
        .collect();
    db.add_batch(&vecs, &metas).unwrap();
    db
}

#[test]
fn search_where_matches_the_id_set_api() {
    let dir = TempDir::new().unwrap();
    let db = hnsw_db(&dir);
    let filter = Filter::from_json(r#"{"meta.bucket": {"$lt": 5}, "meta.lang": "en"}"#).unwrap();
    let allowed = db.ids_where(|m| filter.matches(m)).unwrap();
    assert_eq!(allowed.len(), 36); // even ids with i % 100 in {0, 2, 4}
    for seed in 5000..5010 {
        let q = Vector::new(values(seed));
        let ids = |r: Vec<srvdb::SearchResult>| r.into_iter().map(|r| r.id).collect::<Vec<_>>();
        let got = ids(db.search_where(&q, 10, &filter).unwrap());
        assert_eq!(got.len(), 10);
        assert!(got.iter().all(|id| allowed.contains(id)));
        assert_eq!(got, ids(db.search_filtered(&q, 10, &allowed).unwrap()));
    }
}

#[test]
fn batch_runs_each_query_with_the_same_filter() {
    let dir = TempDir::new().unwrap();
    let db = hnsw_db(&dir);
    let filter = Filter::from_json(r#"{"meta.bucket": {"$in": [1, 2, 3]}}"#).unwrap();
    let queries: Vec<Vector> = (0..3).map(|s| Vector::new(values(9000 + s))).collect();
    let batch = db.search_batch_where(&queries, 5, &filter).unwrap();
    assert_eq!(batch.len(), 3);
    for (q, got) in queries.iter().zip(batch) {
        let single = db.search_where(q, 5, &filter).unwrap();
        let a: Vec<u64> = got.iter().map(|r| r.id).collect();
        let b: Vec<u64> = single.iter().map(|r| r.id).collect();
        assert_eq!(a, b);
    }
}

#[test]
fn fewer_matches_than_k_returns_all_matches() {
    let dir = TempDir::new().unwrap();
    let db = hnsw_db(&dir);
    let filter = Filter::from_json(r#"{"meta.bucket": 7, "meta.lang": "de"}"#).unwrap();
    let got = db
        .search_where(&Vector::new(values(1)), 50, &filter)
        .unwrap();
    assert_eq!(got.len(), 12);
    let none = Filter::from_json(r#"{"meta.bucket": 1000}"#).unwrap();
    assert!(db
        .search_where(&Vector::new(values(1)), 5, &none)
        .unwrap()
        .is_empty());
}

#[test]
fn bad_queries_and_filters_are_invalid_input() {
    let dir = TempDir::new().unwrap();
    let db = hnsw_db(&dir);
    let filter = Filter::from_json(r#"{"meta.bucket": 1}"#).unwrap();
    let q = Vector::new(values(1));
    assert!(matches!(
        db.search_where(&q, 0, &filter),
        Err(Error::InvalidInput(_))
    ));
    let short = Vector::new(vec![0.5; DIM - 1]);
    assert!(matches!(
        db.search_batch_where(&[q, short], 3, &filter),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        Filter::from_json(r#"{"a": {"$like": 1}}"#),
        Err(Error::InvalidInput(_))
    ));
}
