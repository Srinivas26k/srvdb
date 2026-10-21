//! Regression tests: every mode accepted by `new_with_config` (what the
//! Python `mode=` argument uses) must build working storage, and reopen.

use srvdb::types::{DatabaseConfig, IndexType, QuantizationMode};
use srvdb::{SrvDB, Vector, VectorEngine};
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

fn config(index_type: IndexType, quant: Option<QuantizationMode>) -> DatabaseConfig {
    let mut config = DatabaseConfig::new(DIM).unwrap();
    config.index_type = index_type;
    if let Some(mode) = quant {
        config.quantization.enabled = true;
        config.quantization.mode = mode;
    }
    config
}

fn open(dir: &TempDir, config: DatabaseConfig) -> SrvDB {
    SrvDB::new_with_config(dir.path().to_str().unwrap(), config).unwrap()
}

#[test]
fn sq8_mode_trains_on_first_batch_and_reopens() {
    let dir = TempDir::new().unwrap();
    let cfg = || config(IndexType::ScalarQuantized, Some(QuantizationMode::Scalar));
    let vectors = random_vectors(500);

    {
        let mut db = open(&dir, cfg());
        assert!(db.search(&vectors[0], 5).unwrap().is_empty());
        db.add_batch(&vectors, &metas(500)).unwrap();
        db.persist().unwrap();
        assert_eq!(db.search(&vectors[3], 1).unwrap()[0].id, 3);
    }

    let db = open(&dir, cfg());
    assert_eq!(db.count(), 500);
    assert_eq!(db.search(&vectors[3], 1).unwrap()[0].id, 3);
    assert!(!dir.path().join("vectors.bin").exists());
}

#[test]
fn pq_mode_trains_on_first_batch_and_reopens() {
    let dir = TempDir::new().unwrap();
    let cfg = || config(IndexType::ProductQuantized, Some(QuantizationMode::Product));
    let vectors = random_vectors(300);

    {
        let mut db = open(&dir, cfg());
        db.add_batch(&vectors, &metas(300)).unwrap();
        db.persist().unwrap();
        assert_eq!(db.search(&vectors[0], 5).unwrap().len(), 5);
    }

    let db = open(&dir, cfg());
    assert_eq!(db.count(), 300);
    assert_eq!(db.search(&vectors[0], 5).unwrap().len(), 5);
}

#[test]
fn pq_mode_rejects_too_small_first_batch() {
    let dir = TempDir::new().unwrap();
    let mut db = open(
        &dir,
        config(IndexType::ProductQuantized, Some(QuantizationMode::Product)),
    );
    let vectors = random_vectors(10);

    let err = db.add_batch(&vectors, &metas(10)).unwrap_err();
    assert!(err.to_string().contains("at least 256"));
}

#[test]
fn hnsw_mode_builds_graph_and_reopens() {
    let dir = TempDir::new().unwrap();
    let vectors = random_vectors(600);

    {
        let mut db = open(&dir, config(IndexType::HNSW, None));
        db.add_batch(&vectors[..500], &metas(500)).unwrap();
        db.persist().unwrap();
        let graph = dir.path().join("hnsw.graph");
        let older_graph = std::fs::read(&graph).unwrap();
        db.add_batch(&vectors[500..], &metas(100)).unwrap();
        db.persist().unwrap();
        drop(db);
        // An index file older than the committed vectors (indexes are
        // derived data): the missing 100 must be indexed on reopen
        std::fs::write(&graph, older_graph).unwrap();
    }

    let db = open(&dir, config(IndexType::Flat, None));
    assert_eq!(db.current_mode, srvdb::IndexMode::Hnsw);
    assert!(dir.path().join("hnsw.graph").exists());
    for i in [0, 250, 499, 550, 599] {
        assert_eq!(db.search(&vectors[i], 1).unwrap()[0].id, i as u64);
    }
}
