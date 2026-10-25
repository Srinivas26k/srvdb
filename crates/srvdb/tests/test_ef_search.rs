//! set_ef_search changes HNSW search quality (it used to be ignored)

use srvdb::{HNSWConfig, SrvDB, Vector, VectorEngine};
use std::collections::HashSet;
use tempfile::TempDir;

const DIM: usize = 128;
const N: u64 = 3000;

fn vector(seed: u64) -> Vector {
    let mut state = (seed + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    Vector::new(
        (0..DIM)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect(),
    )
}

/// Mean recall@10 of the HNSW database against exact search
fn recall(hnsw: &SrvDB, exact: &SrvDB) -> f32 {
    let queries: Vec<Vector> = (0..50).map(|i| vector(N + 1000 + i)).collect();
    let mut hits = 0;
    for q in &queries {
        let truth: HashSet<u64> = exact.search(q, 10).unwrap().iter().map(|r| r.id).collect();
        hits += hnsw
            .search(q, 10)
            .unwrap()
            .iter()
            .filter(|r| truth.contains(&r.id))
            .count();
    }
    hits as f32 / (queries.len() * 10) as f32
}

#[test]
fn higher_ef_search_gives_higher_recall() {
    let (hdir, edir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let config = HNSWConfig::new(8, 40, 10);
    let mut hnsw = SrvDB::new_with_hnsw(hdir.path().to_str().unwrap(), DIM, config).unwrap();
    let mut exact = SrvDB::new_with_config(
        edir.path().to_str().unwrap(),
        srvdb::DatabaseConfig::new(DIM).unwrap(),
    )
    .unwrap();
    let vectors: Vec<Vector> = (0..N).map(vector).collect();
    let metas: Vec<String> = (0..N).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    hnsw.add_batch(&vectors, &metas).unwrap();
    exact.add_batch(&vectors, &metas).unwrap();

    hnsw.set_ef_search(10).unwrap();
    let low = recall(&hnsw, &exact);
    hnsw.set_ef_search(300).unwrap();
    let high = recall(&hnsw, &exact);
    assert!(
        high > low + 0.05,
        "recall ef=10: {low:.3}, ef=300: {high:.3}"
    );
    assert!(high > 0.9, "recall ef=300: {high:.3}");

    // The setting is saved with the graph
    hnsw.persist().unwrap();
    drop(hnsw);
    let reopened =
        SrvDB::new_with_hnsw(hdir.path().to_str().unwrap(), DIM, HNSWConfig::default()).unwrap();
    assert!((recall(&reopened, &exact) - high).abs() < 1e-6);
}

#[test]
fn ef_search_must_be_positive() {
    let dir = TempDir::new().unwrap();
    let mut db =
        SrvDB::new_with_hnsw(dir.path().to_str().unwrap(), DIM, HNSWConfig::default()).unwrap();
    assert!(matches!(
        db.set_ef_search(0),
        Err(srvdb::Error::InvalidInput(_))
    ));
}
