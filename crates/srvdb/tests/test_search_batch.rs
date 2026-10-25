//! search_batch uses the same index as search (it used to scan flat)

use srvdb::types::{DatabaseConfig, IndexType};
use srvdb::{IVFConfig, IndexMode, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;

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

fn filled(dir: &TempDir, index: IndexType) -> SrvDB {
    let mut config = DatabaseConfig::new(DIM).unwrap();
    config.index_type = index;
    let mut db = SrvDB::new_with_config(dir.path().to_str().unwrap(), config).unwrap();
    let vectors: Vec<Vector> = (0..1500).map(vector).collect();
    let metas: Vec<String> = (0..1500).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    db.add_batch(&vectors, &metas).unwrap();
    db
}

fn ids(results: &[srvdb::SearchResult]) -> Vec<u64> {
    results.iter().map(|r| r.id).collect()
}

fn assert_batch_matches_single(db: &SrvDB) {
    let queries: Vec<Vector> = (5000..5040).map(vector).collect();
    let batch = db.search_batch(&queries, 10).unwrap();
    for (q, from_batch) in queries.iter().zip(&batch) {
        let single = db.search(q, 10).unwrap();
        assert_eq!(ids(from_batch), ids(&single));
        assert_eq!(from_batch[0].metadata, single[0].metadata);
    }
}

#[test]
fn batch_uses_hnsw() {
    let dir = TempDir::new().unwrap();
    let mut db = filled(&dir, IndexType::HNSW);
    db.set_ef_search(10).unwrap(); // approximate, so a flat scan would differ
    assert_batch_matches_single(&db);
}

#[test]
fn batch_uses_ivf() {
    let dir = TempDir::new().unwrap();
    let mut db = filled(&dir, IndexType::Flat);
    db.configure_ivf(IVFConfig {
        nlist: 8,
        nprobe: 1, // approximate, so a flat scan would differ
        ..Default::default()
    })
    .unwrap();
    db.set_mode(IndexMode::Ivf);
    db.train_ivf().unwrap();
    assert_batch_matches_single(&db);
}
