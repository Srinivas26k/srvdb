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

/// postcard varint of `v` (the encoding of the settings payload)
fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// A well-formed, correctly checksummed settings file with any values:
/// what an attacker could write. Fields in payload order.
fn crafted(kind: u8, dimension: u64, hnsw: [u64; 3], ivf: [u64; 2]) -> Vec<u8> {
    let mut payload = Vec::new();
    varint(1, &mut payload); // payload format
    payload.push(kind);
    for v in [dimension, hnsw[0], hnsw[1], hnsw[2], ivf[0], ivf[1]] {
        varint(v, &mut payload);
    }
    payload.extend_from_slice(&[0, 0, 0]); // RaBitQ bits, rerank, seed
    let mut file = b"SRVDIDXS".to_vec();
    file.extend_from_slice(&2u16.to_le_bytes());
    file.extend_from_slice(&[0, 0]);
    file.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    file.extend_from_slice(&crc32c::crc32c(&payload).to_le_bytes());
    file.extend_from_slice(&payload);
    file
}

/// Huge values in a valid-looking file are refused before anything is
/// allocated from them (a million IVF partitions, m = 4096, dimension 65535)
#[test]
fn crafted_settings_with_huge_values_are_corrupt() {
    let dir = TempDir::new().unwrap();
    hnsw_db(dir.path());
    let path = dir.path().join(FILE);
    // Sanity: the hand-built encoding is accepted when values are in range
    std::fs::write(&path, crafted(1, 128, [8, 64, 40], [0, 0])).unwrap();
    let db = SrvDB::open_existing(dir.path().to_str().unwrap()).unwrap();
    assert_eq!(db.count(), 50);
    drop(db);

    let cases = [
        crafted(5, 128, [16, 200, 50], [1_000_000, 10]),
        crafted(5, 128, [16, 200, 50], [100, 1_000_000]),
        crafted(1, 128, [4096, 200, 50], [0, 0]),
        crafted(1, 128, [16, 1_000_000, 50], [0, 0]),
        crafted(1, 65_535, [16, 200, 50], [0, 0]),
        crafted(0, 4097, [0, 0, 0], [0, 0]),
    ];
    for bytes in cases {
        std::fs::write(&path, bytes).unwrap();
        let err = open_err(dir.path());
        assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
    }
}

#[test]
fn configure_ivf_and_hnsw_settings_are_bounded() {
    use srvdb::IVFConfig;
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let mut db = SrvDB::new_with_config(path, DatabaseConfig::new(DIM).unwrap()).unwrap();
    let ivf = |edit: fn(&mut IVFConfig)| {
        let mut config = IVFConfig::default();
        edit(&mut config);
        config
    };
    let bad = [
        ivf(|c| c.nlist = 0),
        ivf(|c| c.nlist = 1_000_000),
        ivf(|c| {
            c.nlist = 8;
            c.nprobe = 100_000;
        }),
        ivf(|c| c.nprobe = 0),
        ivf(|c| c.max_iterations = 0),
        ivf(|c| c.tolerance = f32::NAN),
        ivf(|c| c.hnsw_config = HNSWConfig::new(4096, 200, 50)),
    ];
    for config in bad {
        let err = db.configure_ivf(config).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    }
    assert!(db.configure_ivf(IVFConfig::default()).is_ok());
    let err = db.set_ef_search(10_001).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    drop(db);
    let bad_graphs = [
        HNSWConfig::new(1, 200, 50),
        HNSWConfig::new(16, 0, 50),
        HNSWConfig::new(16, 200, 20_000),
    ];
    for config in bad_graphs {
        let err = SrvDB::new_with_hnsw(path, DIM, config).err().unwrap();
        assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    }
}

/// The settings are read under the lock: a busy database reports Locked
#[test]
fn default_open_of_a_busy_database_is_locked() {
    let dir = TempDir::new().unwrap();
    hnsw_db(dir.path());
    let path = dir.path().to_str().unwrap();
    let _held = SrvDB::open_existing(path).unwrap();
    let err = SrvDB::new_with_config(path, DatabaseConfig::new(DIM).unwrap()).err();
    assert!(matches!(err, Some(Error::Locked(_))), "{err:?}");
}
