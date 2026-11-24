//! RaBitQ mode: resource limits, saving rebuilt codes, and which open
//! errors count as damage (ADR 0008)

use srvdb::{Error, RabitqConfig, SrvDB, Vector, VectorEngine};
use std::path::Path;
use tempfile::TempDir;

const DIM: usize = 128;

fn vectors(n: usize) -> Vec<Vector> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            Vector::new(
                (0..DIM)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                    })
                    .collect(),
            )
        })
        .collect()
}

fn open(dir: &Path, config: RabitqConfig) -> srvdb::Result<SrvDB> {
    SrvDB::new_rabitq(dir.to_str().unwrap(), DIM, config)
}

fn filled(dir: &Path, config: RabitqConfig, n: usize) {
    let mut db = open(dir, config).unwrap();
    db.add_batch(&vectors(n), &vec!["{}".to_string(); n])
        .unwrap();
    db.persist().unwrap();
}

/// The dense rotation is dim^2 values and O(dim^3) work on every open,
/// so RaBitQ refuses dimensions above the database-wide maximum of 4096
#[test]
fn dimension_above_4096_is_rejected_before_anything_is_built() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");
    let config = RabitqConfig::new(1).unwrap();
    for dim in [4097, 65535, usize::MAX] {
        let e = SrvDB::new_rabitq(path.to_str().unwrap(), dim, config).err();
        assert!(matches!(e, Some(Error::InvalidInput(_))), "{dim}: {e:?}");
    }
    assert!(!path.exists(), "nothing is created for a refused dimension");
}

/// Header count of a record file (bytes 8..16)
fn header_count(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap();
    u64::from_le_bytes(bytes[8..16].try_into().unwrap())
}

#[test]
fn rebuilt_codes_are_saved_so_the_next_open_does_not_re_encode() {
    let dir = TempDir::new().unwrap();
    let config = RabitqConfig::new(4).unwrap();
    filled(dir.path(), config, 700);
    let codes = dir.path().join("rabitq_codes.bin");
    std::fs::remove_file(&codes).unwrap();

    // A read-only session: the codes are rebuilt on open, never persisted
    drop(open(dir.path(), config).unwrap());
    assert_eq!(header_count(&codes), 700, "rebuilt codes were not flushed");
    let before = std::fs::read(&codes).unwrap();
    drop(open(dir.path(), config).unwrap());
    assert_eq!(
        std::fs::read(&codes).unwrap(),
        before,
        "second open re-encoded"
    );
}

#[test]
fn io_errors_on_code_files_surface_instead_of_a_rebuild() {
    let dir = TempDir::new().unwrap();
    let config = RabitqConfig::new(2).unwrap();
    filled(dir.path(), config, 50);
    // A directory where the parameters file should be: reading it is an
    // I/O error, not proof of damage, so it must not be deleted
    let params = dir.path().join("rabitq_params.bin");
    std::fs::remove_file(&params).unwrap();
    std::fs::create_dir(&params).unwrap();
    std::fs::write(params.join("keep"), b"x").unwrap();
    let e = open(dir.path(), config).err();
    assert!(matches!(e, Some(Error::Io(_))), "{e:?}");
    assert!(params.join("keep").exists());
}
