//! HNSW over PQ-compressed storage: the graph is built from the quantized
//! vectors, searched, saved by persist() and reloaded on open.

use srvdb::{HNSWConfig, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 64;
const N: usize = 400;

/// Clustered data (40 directions plus noise), so PQ codes keep vectors apart
fn vectors() -> Vec<Vector> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let centres: Vec<Vec<f32>> = (0..40)
        .map(|_| (0..DIM).map(|_| next()).collect())
        .collect();
    (0..N)
        .map(|i| Vector::new(centres[i % 40].iter().map(|c| c + 0.05 * next()).collect()))
        .collect()
}

/// Share of queries whose top hit is in the same cluster as the query
fn recall(db: &SrvDB, vectors: &[Vector]) -> f32 {
    let hits = (0..N)
        .step_by(7)
        .filter(|&i| {
            let results = db.search(&vectors[i], 1).unwrap();
            results
                .first()
                .is_some_and(|r| r.id as usize % 40 == i % 40)
        })
        .count();
    hits as f32 / N.div_ceil(7) as f32
}

#[test]
fn hnsw_pq_indexes_searches_and_reopens() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let vectors = vectors();
    let metas: Vec<String> = (0..N).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    {
        let mut db = SrvDB::new_with_hnsw_quantized(path, &vectors, HNSWConfig::default()).unwrap();
        db.add_batch(&vectors, &metas).unwrap();
        assert!(recall(&db, &vectors) >= 0.8, "{}", recall(&db, &vectors));
        db.persist().unwrap();
    }
    assert!(dir.path().join("hnsw.graph").exists());

    let db = SrvDB::new_with_hnsw_quantized(path, &vectors, HNSWConfig::default()).unwrap();
    assert_eq!(db.count(), N as u64);
    assert!(recall(&db, &vectors) >= 0.8, "{}", recall(&db, &vectors));
}
