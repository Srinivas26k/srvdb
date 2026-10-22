//! Delete and replace (ADR 0004): deleted vectors never come back, top-k
//! stays full, and deletes follow the same durability rules as adds.

use srvdb::types::{DatabaseConfig, IndexType, QuantizationMode};
use srvdb::{Error, IVFConfig, IndexMode, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;
const N: usize = 400;

fn vector(seed: usize) -> Vector {
    let mut state = (seed as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
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

fn open(dir: &TempDir, mode: &str) -> SrvDB {
    let mut config = DatabaseConfig::new(DIM).unwrap();
    match mode {
        "hnsw" => config.index_type = IndexType::HNSW,
        "sq8" | "pq" => {
            config.quantization.enabled = true;
            config.quantization.mode = if mode == "sq8" {
                QuantizationMode::Scalar
            } else {
                QuantizationMode::Product
            };
        }
        _ => {}
    }
    SrvDB::new_with_config(dir.path().to_str().unwrap(), config).unwrap()
}

fn filled(dir: &TempDir, mode: &str) -> (SrvDB, Vec<Vector>) {
    let vectors: Vec<Vector> = (0..N).map(vector).collect();
    let metas: Vec<String> = (0..N).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    let mut db = open(dir, mode);
    db.add_batch(&vectors, &metas).unwrap();
    if mode == "ivf" {
        db.configure_ivf(IVFConfig {
            nlist: 4,
            nprobe: 4,
            ..Default::default()
        })
        .unwrap();
        db.set_mode(IndexMode::Ivf);
        db.train_ivf().unwrap();
    }
    (db, vectors)
}

fn check_mode(mode: &str) {
    let dir = TempDir::new().unwrap();
    let (mut db, vectors) = filled(&dir, mode);

    // Delete the 10 nearest neighbours of a query, including itself
    let query = &vectors[7];
    let nearest: Vec<u64> = db.search(query, 10).unwrap().iter().map(|r| r.id).collect();
    for &id in &nearest {
        assert!(db.delete(id).unwrap());
    }

    let results = db.search(query, 10).unwrap();
    assert_eq!(results.len(), 10, "{mode}: top-k must stay full");
    assert!(results.iter().all(|r| !nearest.contains(&r.id)), "{mode}");
    assert!(db.get_metadata(nearest[0]).unwrap().is_none(), "{mode}");
    assert_eq!(db.live_count(), N as u64 - 10, "{mode}");
    assert_eq!(db.count(), N as u64, "{mode}: ids are never reused");
}

#[test]
fn every_mode_skips_deleted_and_keeps_top_k_full() {
    for mode in ["flat", "hnsw", "sq8", "pq", "ivf"] {
        check_mode(mode);
    }
}

#[test]
fn hnsw_returns_k_results_after_deleting_most_of_the_graph() {
    let dir = TempDir::new().unwrap();
    let (mut db, vectors) = filled(&dir, "hnsw");
    for id in 0..N as u64 {
        if id % 10 != 0 {
            db.delete(id).unwrap();
        }
    }
    let results = db.search(&vectors[3], 10).unwrap();
    assert_eq!(results.len(), 10);
    assert!(results.iter().all(|r| r.id % 10 == 0));
}

#[test]
fn deletes_follow_persist_like_adds() {
    let dir = TempDir::new().unwrap();
    {
        let (mut db, _) = filled(&dir, "flat");
        db.delete(1).unwrap();
        db.persist().unwrap();
        db.delete(2).unwrap(); // never persisted
    }
    let db = open(&dir, "flat");
    assert!(db.is_deleted(1));
    assert!(!db.is_deleted(2));
    assert!(db.get_metadata(2).unwrap().is_some());
    assert_eq!(db.live_count(), N as u64 - 1);
}

#[test]
fn delete_rejects_unknown_ids_and_repeats_are_harmless() {
    let dir = TempDir::new().unwrap();
    let (mut db, _) = filled(&dir, "flat");
    assert!(matches!(db.delete(N as u64), Err(Error::InvalidInput(_))));
    assert!(db.delete(5).unwrap());
    assert!(!db.delete(5).unwrap());
}

#[test]
fn replace_moves_the_record_to_a_new_id() {
    let dir = TempDir::new().unwrap();
    let (mut db, vectors) = filled(&dir, "flat");
    let new_id = db.replace(3, &vectors[399], r#"{"i": "new"}"#).unwrap();
    assert_eq!(new_id, N as u64);
    assert!(db.is_deleted(3));
    assert_eq!(
        db.get_metadata(new_id).unwrap().as_deref(),
        Some(r#"{"i": "new"}"#)
    );
    assert_eq!(db.live_count(), N as u64);

    // Bad input changes nothing
    assert!(db.replace(3, &vectors[0], "{}").is_err());
    assert!(db.replace(4, &Vector::new(vec![0.1; 3]), "{}").is_err());
    assert!(!db.is_deleted(4));
}
