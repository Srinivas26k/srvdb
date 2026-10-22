//! Recovery on open (ADR 0003): a database reopens exactly at its last
//! successful persist(); later writes are discarded, never half-present.

use srvdb::types::{DatabaseConfig, QuantizationMode};
use srvdb::{Error, MetadataStore, SrvDB, Vector, VectorEngine, VectorStorage};
use tempfile::TempDir;

const DIM: usize = 128;

/// Deterministic pseudo-random vector (distinct for every seed)
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

fn metas(range: std::ops::Range<usize>) -> Vec<String> {
    range.map(|i| format!(r#"{{"i": {i}}}"#)).collect()
}

fn config(quant: Option<QuantizationMode>) -> DatabaseConfig {
    let mut config = DatabaseConfig::new(DIM).unwrap();
    if let Some(mode) = quant {
        config.quantization.enabled = true;
        config.quantization.mode = mode;
    }
    config
}

fn open(dir: &TempDir, quant: Option<QuantizationMode>) -> SrvDB {
    SrvDB::new_with_config(dir.path().to_str().unwrap(), config(quant)).unwrap()
}

fn check_discards_uncommitted(quant: Option<QuantizationMode>) {
    let dir = TempDir::new().unwrap();
    let vectors: Vec<Vector> = (0..400).map(vector).collect();
    {
        let mut db = open(&dir, quant);
        db.add_batch(&vectors[..300], &metas(0..300)).unwrap();
        db.persist().unwrap();
        // Written but never committed, then the process "dies"
        db.add_batch(&vectors[300..350], &metas(300..350)).unwrap();
    }

    let mut db = open(&dir, quant);
    assert_eq!(db.count(), 300);
    assert!(db.get_metadata(299).unwrap().is_some());
    assert!(db.get_metadata(300).unwrap().is_none());
    assert_eq!(db.all_metadata().unwrap().len(), 300);

    // New writes continue at the committed position with the right data
    let id = db.add(&vectors[399], r#"{"i": 399}"#).unwrap();
    assert_eq!(id, 300);
    assert_eq!(db.search(&vectors[399], 1).unwrap()[0].id, 300);
}

#[test]
fn flat_discards_writes_after_last_persist() {
    check_discards_uncommitted(None);
}

#[test]
fn sq8_discards_writes_after_last_persist() {
    check_discards_uncommitted(Some(QuantizationMode::Scalar));
}

#[test]
fn pq_discards_writes_after_last_persist() {
    check_discards_uncommitted(Some(QuantizationMode::Product));
}

#[test]
fn nothing_persisted_reopens_empty() {
    let dir = TempDir::new().unwrap();
    {
        let mut db = open(&dir, None);
        db.add(&vector(1), "{}").unwrap();
    }
    let db = open(&dir, None);
    assert_eq!(db.count(), 0);
    assert!(db.all_metadata().unwrap().is_empty());
}

#[test]
fn database_without_commit_record_keeps_its_vectors() {
    // A v0.2 database: vectors and metadata on disk, no commit record
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    {
        let mut storage = VectorStorage::new(path, DIM).unwrap();
        storage
            .append_batch(&(0..3).map(|i| vector(i).data).collect::<Vec<_>>())
            .unwrap();
        storage.flush().unwrap();
        let mut store = MetadataStore::new(path).unwrap();
        store
            .set_batch(&(0..3).map(|i| (i, "{}".to_string())).collect::<Vec<_>>())
            .unwrap();
    }
    let db = open(&dir, None);
    assert_eq!(db.count(), 3);
    assert_eq!(db.all_metadata().unwrap().len(), 3);
}

#[test]
fn missing_committed_vectors_are_reported_as_corruption() {
    let dir = TempDir::new().unwrap();
    {
        let mut db = open(&dir, None);
        db.add_batch(&(0..10).map(vector).collect::<Vec<_>>(), &metas(0..10))
            .unwrap();
        db.persist().unwrap();
    }
    // Lose committed data behind SrvDB's back
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("vectors.bin"))
        .unwrap();
    file.set_len(24 + 5 * DIM as u64 * 4).unwrap();

    let err = SrvDB::new_with_config(dir.path().to_str().unwrap(), config(None))
        .err()
        .unwrap();
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
}

#[test]
fn sq8_crash_during_first_training_reopens_untrained() {
    // Killed after sq8_vectors.bin was created but before the quantizer
    // file was written: the database must open and train on the next batch
    let dir = TempDir::new().unwrap();
    let vectors: Vec<Vector> = (0..20).map(vector).collect();
    {
        let mut db = open(&dir, Some(QuantizationMode::Scalar));
        db.add_batch(&vectors[..10], &metas(0..10)).unwrap();
    }
    std::fs::remove_file(dir.path().join("scalar_quantizer.bin")).unwrap();

    let mut db = open(&dir, Some(QuantizationMode::Scalar));
    assert_eq!(db.count(), 0);
    db.add_batch(&vectors[10..], &metas(10..20)).unwrap();
    db.persist().unwrap();
    assert_eq!(db.search(&vectors[15], 1).unwrap()[0].id, 5);
}

#[test]
fn crash_during_database_creation_leaves_it_openable() {
    // States a kill during the very first open can leave behind
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join(".metadata.db.tmp"), b"partial redb file").unwrap();
    std::fs::write(dir.path().join("vectors.bin"), b"").unwrap();

    let mut db = open(&dir, None);
    db.add(&vector(1), "{}").unwrap();
    db.persist().unwrap();
    drop(db);
    assert_eq!(open(&dir, None).count(), 1);
    assert!(!dir.path().join(".metadata.db.tmp").exists());
}
