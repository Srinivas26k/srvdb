//! Regression tests: vectors added through any path must reach the active
//! graph index, and graph searches must rank by distance to the query.

use srvdb::types::DatabaseConfig;
use srvdb::{HNSWConfig, IVFConfig, IndexMode, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;

fn random_vectors(n: usize) -> Vec<Vector> {
    (0..n)
        .map(|_| Vector::new((0..DIM).map(|_| rand::random::<f32>() - 0.5).collect()))
        .collect()
}

fn metas(n: usize) -> Vec<String> {
    (0..n).map(|i| format!(r#"{{"i": {}}}"#, i)).collect()
}

/// Fraction of the first `probes` vectors that find themselves as top-1
fn self_hit_rate(db: &SrvDB, vectors: &[Vector], probes: usize) -> f32 {
    let hits = (0..probes)
        .filter(|&i| db.search(&vectors[i], 1).unwrap().first().map(|r| r.id) == Some(i as u64))
        .count();
    hits as f32 / probes as f32
}

#[test]
fn hnsw_indexes_add_batch() {
    let dir = TempDir::new().unwrap();
    let mut db =
        SrvDB::new_with_hnsw(dir.path().to_str().unwrap(), DIM, HNSWConfig::default()).unwrap();
    let vectors = random_vectors(1000);

    db.add_batch(&vectors, &metas(1000)).unwrap();

    assert_eq!(db.search(&vectors[0], 10).unwrap().len(), 10);
    assert!(self_hit_rate(&db, &vectors, 50) > 0.95);
}

#[test]
fn hnsw_indexes_single_add() {
    let dir = TempDir::new().unwrap();
    let mut db =
        SrvDB::new_with_hnsw(dir.path().to_str().unwrap(), DIM, HNSWConfig::default()).unwrap();
    let vectors = random_vectors(500);

    for (v, m) in vectors.iter().zip(metas(500)) {
        db.add(v, &m).unwrap();
    }

    assert!(self_hit_rate(&db, &vectors, 50) > 0.95);
}

fn ivf_db(dir: &TempDir, vectors: &[Vector]) -> SrvDB {
    let config = DatabaseConfig::new(DIM).unwrap();
    let mut db = SrvDB::new_with_config(dir.path().to_str().unwrap(), config).unwrap();
    db.add_batch(vectors, &metas(vectors.len())).unwrap();
    db.configure_ivf(IVFConfig {
        nlist: 4,
        nprobe: 4,
        ..Default::default()
    })
    .unwrap();
    db.set_mode(IndexMode::Ivf);
    db.train_ivf().unwrap();
    db
}

#[test]
fn ivf_ranks_by_query_and_honours_config() {
    let dir = TempDir::new().unwrap();
    let vectors = random_vectors(1000);
    let db = ivf_db(&dir, &vectors);

    assert!(self_hit_rate(&db, &vectors, 50) > 0.95);

    // Scores are similarities (higher is better), like other modes
    let results = db.search(&vectors[7], 5).unwrap();
    assert!(results[0].score > 0.99);
    assert!(results.windows(2).all(|w| w[0].score >= w[1].score));
}

#[test]
fn ivf_indexes_vectors_added_after_training() {
    let dir = TempDir::new().unwrap();
    let vectors = random_vectors(600);
    let mut db = ivf_db(&dir, &vectors[..500]);

    db.add_batch(&vectors[500..], &metas(100)).unwrap();

    let hits = (500..550)
        .filter(|&i| db.search(&vectors[i], 1).unwrap()[0].id == i as u64)
        .count();
    assert!(hits >= 48);
}
