//! Persisted index settings (ADR 0011): corruption and migration from 0.3.
//! Crash safety is in `test_crash_settings.rs`, its own binary so that its
//! child processes never briefly share a lock file with these tests.

use srvdb::types::{DatabaseConfig, IndexType};
use srvdb::{Error, HNSWConfig, IndexMode, SrvDB, Vector, VectorEngine};
use std::path::Path;
use tempfile::TempDir;

const DIM: usize = 128;
const FILE: &str = "index_settings.bin";

fn vector(i: u64) -> Vector {
    Vector::new(
        (0..DIM)
            .map(|j| ((i as usize * 7 + j * 13) % 29) as f32 - 14.0 + i as f32 * 0.01)
            .collect(),
    )
}

fn hnsw_db(dir: &Path) {
    let mut db =
        SrvDB::new_with_hnsw(dir.to_str().unwrap(), DIM, HNSWConfig::new(8, 64, 40)).unwrap();
    let vectors: Vec<Vector> = (0..50).map(vector).collect();
    db.add_batch(&vectors, &vec!["{}".to_string(); 50]).unwrap();
    db.persist().unwrap();
}

fn open_err(dir: &Path) -> Error {
    SrvDB::open_existing(dir.to_str().unwrap()).err().unwrap()
}

#[test]
fn damaged_settings_fail_with_checksum_mismatch_or_corrupt() {
    let dir = TempDir::new().unwrap();
    hnsw_db(dir.path());
    let file = dir.path().join(FILE);
    let good = std::fs::read(&file).unwrap();

    let mut flipped = good.clone();
    *flipped.last_mut().unwrap() ^= 0x01;
    std::fs::write(&file, &flipped).unwrap();
    assert!(matches!(open_err(dir.path()), Error::ChecksumMismatch(_)));
    // Every constructor reads the settings, not only open_existing
    let config = DatabaseConfig::new(DIM).unwrap();
    let err = SrvDB::new_with_config(dir.path().to_str().unwrap(), config).err();
    assert!(matches!(err, Some(Error::ChecksumMismatch(_))), "{err:?}");

    std::fs::write(&file, &good[..good.len() - 3]).unwrap();
    assert!(matches!(open_err(dir.path()), Error::Corrupt(_)));

    let mut magic = good.clone();
    magic[..4].copy_from_slice(b"JUNK");
    std::fs::write(&file, &magic).unwrap();
    assert!(matches!(open_err(dir.path()), Error::Corrupt(_)));

    let mut version = good.clone();
    version[8] = 0x7F;
    std::fs::write(&file, &version).unwrap();
    assert!(matches!(open_err(dir.path()), Error::Corrupt(_)));

    std::fs::write(&file, &good).unwrap();
    assert_eq!(
        SrvDB::open_existing(dir.path().to_str().unwrap())
            .unwrap()
            .count(),
        50
    );
}

#[test]
fn missing_or_empty_directories_are_not_databases() {
    let dir = TempDir::new().unwrap();
    let missing = dir.path().join("missing");
    assert!(matches!(open_err(&missing), Error::InvalidInput(_)));
    assert!(matches!(open_err(dir.path()), Error::InvalidInput(_)));
    assert!(!missing.exists(), "open_existing must not create anything");
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

/// A 0.3 database has no settings: it opens as before, and the settings
/// are recorded the first time it is opened with an explicit mode.
#[test]
fn database_from_0_3_records_its_settings_on_first_explicit_open() {
    let dir = copy_fixture("hnsw");
    let path = dir.path().to_str().unwrap();
    {
        // As before: flat by default (the existing graph is still used),
        // and a default open does not pin the database to a mode
        let db = SrvDB::open_existing(path).unwrap();
        assert_eq!(db.count(), 40);
        assert_eq!(db.search(&vector(0), 1).unwrap().len(), 1);
    }
    assert!(!dir.path().join(FILE).exists());
    {
        let mut config = DatabaseConfig::new(DIM).unwrap();
        config.index_type = IndexType::HNSW;
        let db = SrvDB::new_with_config(path, config).unwrap();
        assert_eq!(db.count(), 40);
    }
    assert!(dir.path().join(FILE).exists());
    let db = SrvDB::open_existing(path).unwrap();
    assert_eq!(db.current_mode, IndexMode::Hnsw);
    assert_eq!(db.count(), 40);
    assert!(db.get_metadata(39).unwrap().is_some());
}

#[test]
fn quantized_database_from_0_3_must_be_opened_with_its_mode_once() {
    let dir = copy_fixture("sq8");
    assert!(matches!(open_err(dir.path()), Error::InvalidState(_)));
    assert!(!dir.path().join(FILE).exists());
}
