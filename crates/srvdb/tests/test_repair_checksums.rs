//! `SrvDB::rebuild_checksums` (issue #33, ADR 0007): a missing or damaged
//! `.crc` sidecar is recomputed on explicit request, and an intact sidecar
//! that disagrees with the records is never overwritten.

use srvdb::{CreateOptions, Error, IndexChoice, SrvDB, Vector, VectorEngine, VectorStorage};
use std::path::Path;
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

/// A persisted database with N vectors
fn persisted(dir: &Path, index: IndexChoice) {
    let mut options = CreateOptions::new(index);
    if index == IndexChoice::Pq {
        options
            .set_training((0..N).map(|i| vector(i).data).collect())
            .unwrap();
    }
    let mut db = SrvDB::create(dir.to_str().unwrap(), DIM, &options).unwrap();
    let vectors: Vec<Vector> = (0..N).map(vector).collect();
    let metas: Vec<String> = (0..N).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    db.add_batch(&vectors, &metas).unwrap();
    db.persist().unwrap();
}

fn reopen(dir: &Path) -> srvdb::Result<SrvDB> {
    SrvDB::open_existing(dir.to_str().unwrap())
}

fn rebuild(dir: &Path) -> srvdb::Result<Vec<String>> {
    SrvDB::rebuild_checksums(dir.to_str().unwrap())
}

/// Ids of the top 5 hits for a few queries
fn answers(db: &SrvDB) -> Vec<Vec<u64>> {
    (0..5)
        .map(|q| {
            let hits = db.search(&vector(q * 37), 5).unwrap();
            hits.iter().map(|h| h.id).collect()
        })
        .collect()
}

fn sidecars(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "crc"))
        .collect();
    found.sort();
    found
}

fn flip(path: &Path, at: usize) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[at] ^= 0x04;
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn missing_sidecars_are_rebuilt_and_the_data_reopens_unchanged() {
    for index in [
        IndexChoice::Flat,
        IndexChoice::Sq8,
        IndexChoice::Pq,
        IndexChoice::Rabitq,
    ] {
        let dir = TempDir::new().unwrap();
        persisted(dir.path(), index);
        let before = answers(&reopen(dir.path()).unwrap());
        let removed = sidecars(dir.path());
        assert!(!removed.is_empty(), "{index}");
        for side in &removed {
            std::fs::remove_file(side).unwrap();
        }
        let err = reopen(dir.path()).err().unwrap();
        assert!(matches!(err, Error::Corrupt(_)), "{index}: {err:?}");

        let rebuilt = rebuild(dir.path()).unwrap();
        assert_eq!(rebuilt.len(), removed.len(), "{index}: {rebuilt:?}");
        assert_eq!(sidecars(dir.path()), removed, "{index}");
        let db = reopen(dir.path()).unwrap();
        assert_eq!(db.count(), N, "{index}");
        assert_eq!(answers(&db), before, "{index}");
        assert_eq!(
            db.get_metadata(7).unwrap().as_deref(),
            Some(r#"{"i": 7}"#),
            "{index}"
        );
    }
}

#[test]
fn damaged_sidecar_is_rebuilt() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), IndexChoice::Flat);
    let side = dir.path().join("vectors.bin.crc");
    let len = std::fs::metadata(&side).unwrap().len() as usize;
    flip(&side, len - 1);
    assert!(reopen(dir.path()).is_err());
    assert_eq!(rebuild(dir.path()).unwrap(), ["vectors.bin.crc"]);
    assert_eq!(reopen(dir.path()).unwrap().count(), N);
}

#[test]
fn damaged_records_with_an_intact_sidecar_are_refused() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), IndexChoice::Flat);
    let side = dir.path().join("vectors.bin.crc");
    let sidecar = std::fs::read(&side).unwrap();
    flip(&dir.path().join("vectors.bin"), 24 + 1000);

    let err = rebuild(dir.path()).unwrap_err();
    assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");
    assert!(err.to_string().contains("backup"), "{err}");
    assert_eq!(std::fs::read(&side).unwrap(), sidecar, "sidecar rewritten");
    let err = reopen(dir.path()).err().unwrap();
    assert!(matches!(err, Error::ChecksumMismatch(_)), "{err:?}");
}

#[test]
fn intact_database_is_left_alone() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), IndexChoice::Sq8);
    let before: Vec<Vec<u8>> = sidecars(dir.path())
        .iter()
        .map(|p| std::fs::read(p).unwrap())
        .collect();
    assert!(rebuild(dir.path()).unwrap().is_empty());
    let after: Vec<Vec<u8>> = sidecars(dir.path())
        .iter()
        .map(|p| std::fs::read(p).unwrap())
        .collect();
    assert_eq!(before, after);
}

#[test]
fn open_database_and_bad_paths_are_refused() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), IndexChoice::Flat);
    let db = reopen(dir.path()).unwrap();
    assert!(matches!(rebuild(dir.path()), Err(Error::Locked(_))));
    drop(db);
    let missing = dir.path().join("nope");
    assert!(matches!(rebuild(&missing), Err(Error::NotFound(_))));
}

#[test]
fn truncated_vector_file_is_refused_when_the_sidecar_is_intact() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), IndexChoice::Flat);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("vectors.bin"))
        .unwrap();
    file.set_len(24 + 100 * DIM as u64 * 4).unwrap();
    drop(file);
    assert!(matches!(rebuild(dir.path()), Err(Error::Corrupt(_))));
}

/// Crash state: a persist() died after the vector file fsync (header says
/// 350 records) but before the redb commit (300 committed), a rebuild
/// crashed before its rename (stale temp file), and the sidecar was lost.
/// The rebuild covers the 350 records; open still recovers to 300.
#[test]
fn rebuild_after_crashes_recovers_to_the_committed_count() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), IndexChoice::Flat);
    {
        let mut storage = VectorStorage::new(dir.path().to_str().unwrap(), DIM).unwrap();
        for id in N..N + 50 {
            storage.append(&vector(id).data).unwrap();
        }
        storage.flush().unwrap();
    }
    std::fs::remove_file(dir.path().join("vectors.bin.crc")).unwrap();
    std::fs::write(dir.path().join(".vectors.bin.crc.tmp"), b"partial").unwrap();

    assert_eq!(rebuild(dir.path()).unwrap(), ["vectors.bin.crc"]);
    let mut db = reopen(dir.path()).unwrap();
    assert_eq!(db.count(), N);
    assert_eq!(db.search(&vector(299), 1).unwrap()[0].id, 299);
    db.add(&vector(N), "{}").unwrap();
    db.persist().unwrap();
    drop(db);
    assert_eq!(reopen(dir.path()).unwrap().count(), N + 1);
}

/// Damage one byte of a vector file header (the sidecar stays intact);
/// the rebuild must refuse with the header message and leave the sidecar
fn assert_header_damage_refused(index: IndexChoice, file: &str, at: usize, damage: fn(u8) -> u8) {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), index);
    let side = dir.path().join(format!("{file}.crc"));
    let before = std::fs::read(&side).unwrap();
    let path = dir.path().join(file);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[at] = damage(bytes[at]);
    std::fs::write(&path, bytes).unwrap();
    let err = rebuild(dir.path()).unwrap_err();
    assert!(matches!(err, Error::Corrupt(_)), "{file}@{at}: {err:?}");
    assert_eq!(
        std::fs::read(&side).unwrap(),
        before,
        "{file}: sidecar rewritten"
    );
}

#[test]
fn flipped_dimension_bit_with_an_intact_sidecar_is_refused() {
    // Dimension is bytes 6..8: 128 -> 129 and 128 -> 384
    assert_header_damage_refused(IndexChoice::Flat, "vectors.bin", 6, |b| b ^ 0x01);
    assert_header_damage_refused(IndexChoice::Flat, "vectors.bin", 7, |b| b ^ 0x01);
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), IndexChoice::Flat);
    flip(&dir.path().join("vectors.bin"), 6);
    let err = rebuild(dir.path()).unwrap_err().to_string();
    assert!(
        err.contains("header or codebooks disagree with an intact checksum file"),
        "{err}"
    );
}

#[test]
fn flipped_rabitq_mode_byte_is_refused() {
    // Mode byte 16 is 0x10 | bits: another supported bit width (a valid
    // looking header with a different record size), and a foreign mode
    let other_width = |b: u8| if b & 0x0F == 1 { 0x12 } else { 0x11 };
    assert_header_damage_refused(IndexChoice::Rabitq, "rabitq_codes.bin", 16, other_width);
    assert_header_damage_refused(IndexChoice::Rabitq, "rabitq_codes.bin", 16, |b| b ^ 0x40);
}

#[test]
fn directory_without_vector_files_is_not_found_and_not_locked() {
    let dir = TempDir::new().unwrap();
    let err = rebuild(dir.path()).unwrap_err();
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        0,
        "file left"
    );
}

#[test]
fn vector_file_shorter_than_its_header_is_corrupt() {
    let dir = TempDir::new().unwrap();
    persisted(dir.path(), IndexChoice::Flat);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("vectors.bin"))
        .unwrap();
    file.set_len(10).unwrap();
    drop(file);
    let err = rebuild(dir.path()).unwrap_err();
    assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
}
