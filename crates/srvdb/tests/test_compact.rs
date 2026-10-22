//! Compaction (ADR 0004): deleted vectors leave the disk, ids become dense,
//! and an interrupted compaction is finished or discarded on open.

use srvdb::types::{DatabaseConfig, IndexType, QuantizationMode};
use srvdb::{IVFConfig, IndexMode, MetadataStore, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;
const N: u64 = 300;

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

fn open(dir: &TempDir, mode: &str) -> SrvDB {
    let mut config = DatabaseConfig::new(DIM).unwrap();
    match mode {
        "hnsw" => config.index_type = IndexType::HNSW,
        "sq8" => {
            config.quantization.enabled = true;
            config.quantization.mode = QuantizationMode::Scalar;
        }
        "pq" => {
            config.quantization.enabled = true;
            config.quantization.mode = QuantizationMode::Product;
        }
        _ => {}
    }
    SrvDB::new_with_config(dir.path().to_str().unwrap(), config).unwrap()
}

/// N vectors with metadata {"orig": i}; every third one deleted
fn prepared(dir: &TempDir, mode: &str) -> SrvDB {
    let mut db = open(dir, mode);
    let vectors: Vec<Vector> = (0..N).map(vector).collect();
    let metas: Vec<String> = (0..N).map(|i| format!(r#"{{"orig": {i}}}"#)).collect();
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
    for id in (0..N).step_by(3) {
        db.delete(id).unwrap();
    }
    db
}

/// Original id of the vector now stored at `new_id`
fn orig(db: &SrvDB, new_id: u64) -> u64 {
    let meta = db.get_metadata(new_id).unwrap().unwrap();
    let value: serde_json::Value = serde_json::from_str(&meta).unwrap();
    value["orig"].as_u64().unwrap()
}

fn check_mode(mode: &str) {
    let dir = TempDir::new().unwrap();
    let mut db = prepared(&dir, mode);
    let removed = db.compact().unwrap();
    assert_eq!(removed, N.div_ceil(3), "{mode}");
    let live = N - removed;
    assert_eq!((db.count(), db.live_count()), (live, live), "{mode}");

    // Survivors keep their order; metadata moved with them
    let survivors: Vec<u64> = (0..N).filter(|i| i % 3 != 0).collect();
    for (new_id, &old) in survivors.iter().enumerate().step_by(37) {
        assert_eq!(orig(&db, new_id as u64), old, "{mode}");
        let hit = &db.search(&vector(old), 1).unwrap()[0];
        assert_eq!(hit.id, new_id as u64, "{mode}: vector {old}");
    }
    drop(db);

    let db = open(&dir, mode);
    assert_eq!(db.count(), live, "{mode}: after reopen");
    assert_eq!(orig(&db, 5), survivors[5], "{mode}: after reopen");
}

#[test]
fn compaction_works_in_every_mode() {
    for mode in ["flat", "hnsw", "sq8", "pq", "ivf"] {
        check_mode(mode);
    }
}

#[test]
fn compaction_shrinks_the_file_and_is_a_no_op_without_deletes() {
    let dir = TempDir::new().unwrap();
    let mut db = prepared(&dir, "flat");
    db.compact().unwrap();
    let size = std::fs::metadata(dir.path().join("vectors.bin"))
        .unwrap()
        .len();
    assert_eq!(size, 24 + db.count() * DIM as u64 * 4);
    assert_eq!(db.compact().unwrap(), 0);
}

#[test]
fn abandoned_compaction_is_discarded_on_open() {
    let dir = TempDir::new().unwrap();
    {
        let mut db = prepared(&dir, "flat");
        db.persist().unwrap();
    }
    // Killed after writing the staged file, before the redb commit
    std::fs::write(dir.path().join("vectors.bin.compact"), b"half written").unwrap();

    let db = open(&dir, "flat");
    assert!(!dir.path().join("vectors.bin.compact").exists());
    assert_eq!(db.count(), N);
    assert_eq!(db.live_count(), N - N.div_ceil(3));
}

#[test]
fn committed_compaction_is_finished_on_open() {
    let dir = TempDir::new().unwrap();
    let keep: Vec<u64> = (0..N).filter(|i| i % 3 != 0).collect();
    {
        let mut db = prepared(&dir, "flat");
        db.persist().unwrap();
    }
    // Killed after the redb commit, before the rename. The store is opened
    // first (as in the engine), then the staged file is written by hand
    // (24-byte header, then the kept f32 records), then the commit.
    let mut store = MetadataStore::new(dir.path().to_str().unwrap()).unwrap();
    let original = std::fs::read(dir.path().join("vectors.bin")).unwrap();
    let record = DIM * 4;
    let mut staged = original[..24].to_vec();
    staged[8..16].copy_from_slice(&(keep.len() as u64).to_le_bytes());
    for &id in &keep {
        let at = 24 + id as usize * record;
        staged.extend_from_slice(&original[at..at + record]);
    }
    std::fs::write(dir.path().join("vectors.bin.compact"), staged).unwrap();
    store.compact(&keep).unwrap();
    drop(store);

    let db = open(&dir, "flat");
    assert_eq!(db.count(), keep.len() as u64);
    assert_eq!(db.live_count(), keep.len() as u64);
    assert_eq!(orig(&db, 10), keep[10]);
    assert_eq!(db.search(&vector(keep[10]), 1).unwrap()[0].id, 10);
    assert!(!dir.path().join("vectors.bin.compact").exists());
}
