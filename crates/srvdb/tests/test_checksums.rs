//! Checksums (ADR 0007): damaged vector files fail to open with a typed
//! error, never wrong results; files from SrvDB 0.3 open and are upgraded.
//! Fixtures in tests/data/v0.3 were written by SrvDB 0.3 (vector header
//! version 4, envelope version 1), trimmed to the committed records.

use srvdb::types::{DatabaseConfig, IndexType, QuantizationMode};
use srvdb::{Error, ScalarQuantizedStorage, SrvDB, Vector, VectorEngine};
use std::path::Path;
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

fn config(mode: &str) -> DatabaseConfig {
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
    config
}

fn open(dir: &Path, mode: &str) -> srvdb::Result<SrvDB> {
    SrvDB::new_with_config(dir.to_str().unwrap(), config(mode))
}

/// A persisted database with 300 vectors; returns its vector file name
fn persisted(dir: &Path, mode: &str) -> &'static str {
    let mut db = open(dir, mode).unwrap();
    let vectors: Vec<Vector> = (0..300).map(vector).collect();
    let metas: Vec<String> = (0..300).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    db.add_batch(&vectors, &metas).unwrap();
    db.persist().unwrap();
    match mode {
        "sq8" => "sq8_vectors.bin",
        "pq" => "quantized_vectors.bin",
        _ => "vectors.bin",
    }
}

fn flip(path: &Path, at: usize) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[at] ^= 0x04;
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn flipped_byte_in_vector_file_fails_open() {
    for mode in ["flat", "sq8", "pq"] {
        let dir = TempDir::new().unwrap();
        let file = persisted(dir.path(), mode);
        assert!(open(dir.path(), mode).is_ok(), "{mode}");
        flip(&dir.path().join(file), 24 + 1000);
        let err = open(dir.path(), mode).err().unwrap();
        assert!(matches!(err, Error::ChecksumMismatch(_)), "{mode}: {err:?}");
        assert!(err.to_string().contains(file), "{err}");
    }
}

#[test]
fn checksums_follow_appends_reopen_and_compaction() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), "flat");
    {
        let mut db = open(dir.path(), "flat").unwrap();
        db.add(&vector(300), r#"{"i": 300}"#).unwrap();
        db.persist().unwrap();
        db.add(&vector(301), r#"{"i": 301}"#).unwrap(); // never persisted
    }
    {
        let mut db = open(dir.path(), "flat").unwrap();
        assert_eq!(db.count(), 301);
        db.delete(5).unwrap();
        db.compact().unwrap();
    }
    let db = open(dir.path(), "flat").unwrap();
    assert_eq!(db.count(), 300);
    assert_eq!(db.search(&vector(300), 1).unwrap()[0].id, 299);
}

#[test]
fn missing_or_damaged_checksum_file_fails_open() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), "flat");
    let side = dir.path().join("vectors.bin.crc");
    let len = std::fs::metadata(&side).unwrap().len() as usize;
    flip(&side, len - 1);
    let err = open(dir.path(), "flat").err().unwrap();
    assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");

    std::fs::remove_file(&side).unwrap();
    let err = open(dir.path(), "flat").err().unwrap();
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");

    // Documented recovery (website durability guide): mark the vector file
    // as version 4 and the next open rebuilds the checksums
    let vectors = dir.path().join("vectors.bin");
    let mut bytes = std::fs::read(&vectors).unwrap();
    bytes[4..6].copy_from_slice(&4u16.to_le_bytes());
    std::fs::write(&vectors, bytes).unwrap();
    assert_eq!(open(dir.path(), "flat").unwrap().count(), 300);
    assert!(side.exists());
    assert_eq!(open(dir.path(), "flat").unwrap().count(), 300);
}

#[test]
fn truncated_vector_file_fails_open() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), "flat");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("vectors.bin"))
        .unwrap();
    file.set_len(24 + 100 * DIM as u64 * 4).unwrap();
    drop(file);
    let err = open(dir.path(), "flat").err().unwrap();
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
}

#[test]
fn damaged_hnsw_graph_is_rebuilt_from_verified_vectors() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), "hnsw");
    let graph = dir.path().join("hnsw.graph");
    flip(
        &graph,
        std::fs::metadata(&graph).unwrap().len() as usize - 1,
    );
    let db = open(dir.path(), "hnsw").unwrap();
    assert_eq!(db.count(), 300);
    assert_eq!(db.search(&vector(42), 1).unwrap()[0].id, 42);
}

fn copy_fixture(name: &str) -> TempDir {
    let dir = TempDir::new().unwrap();
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/v0.3")
        .join(name);
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dir.path().join(entry.file_name())).unwrap();
    }
    dir
}

fn header_version(path: &Path) -> u16 {
    let bytes = std::fs::read(path).unwrap();
    u16::from_le_bytes([bytes[4], bytes[5]])
}

#[test]
fn database_from_0_3_opens_and_is_upgraded() {
    let dir = copy_fixture("hnsw");
    let vectors = dir.path().join("vectors.bin");
    assert_eq!(header_version(&vectors), 4);
    let before = std::fs::read(&vectors).unwrap();
    let first: Vec<f32> = before[24..24 + DIM * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    {
        let mut db = open(dir.path(), "hnsw").unwrap();
        assert_eq!(db.count(), 40);
        assert_eq!(db.search(&Vector::new(first.clone()), 1).unwrap()[0].id, 0);
        assert!(db.get_metadata(39).unwrap().is_some());
        db.persist().unwrap();
    }
    assert_eq!(header_version(&vectors), 5);
    assert!(dir.path().join("vectors.bin.crc").exists());
    // persist() rewrote the graph in envelope version 2 (CRC32C)
    let graph = std::fs::read(dir.path().join("hnsw.graph")).unwrap();
    assert_eq!(u16::from_le_bytes([graph[8], graph[9]]), 2);

    // The upgraded file reopens and its checksums now guard the records
    let mut db = open(dir.path(), "hnsw").unwrap();
    assert_eq!(db.count(), 40);
    db.add(&vector(1), r#"{"i": 40}"#).unwrap();
    db.persist().unwrap();
    drop(db);
    flip(&vectors, 24 + 7);
    let err = open(dir.path(), "hnsw").err().unwrap();
    assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");
}

#[test]
fn sq8_storage_from_0_3_opens_and_is_upgraded() {
    let dir = copy_fixture("sq8");
    let path = dir.path().to_str().unwrap();
    {
        let storage = ScalarQuantizedStorage::load(path, DIM).unwrap();
        assert_eq!(storage.count(), 40);
        assert_eq!(storage.get(39).unwrap().len(), DIM);
    }
    assert_eq!(header_version(&dir.path().join("sq8_vectors.bin")), 5);
    let storage = ScalarQuantizedStorage::load(path, DIM).unwrap();
    assert_eq!(storage.count(), 40);
}
