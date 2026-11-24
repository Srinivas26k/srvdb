//! Crash tests for the checksum sidecar ordering (ADR 0007).
//!
//! `persist()` runs: vector file write + fsync, then sidecar replacement
//! (temp file, fsync, rename), then the redb commit. A random kill rarely
//! lands between those steps, so this test re-runs itself as a child that
//! dies with `std::process::abort` (no destructors, no flush) at an exact
//! point, leaving the files as a kill there would:
//!
//! - `fsync`: after the vector file fsync, before the sidecar rename (the
//!   new table sits in the temp file, the old sidecar is still in place)
//! - `sidecar`: after the sidecar rename, before the redb commit
//!
//! The parent then reopens and checks that exactly the committed data is
//! there, intact, and that the database keeps working.

use srvdb::types::DatabaseConfig;
use srvdb::{SrvDB, Vector, VectorEngine, VectorStorage};
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

const DIM: usize = 128;
const COMMITTED: u64 = 300;
const CHILD_DIR: &str = "SRVDB_CRC_CRASH_DIR";
const CHILD_POINT: &str = "SRVDB_CRC_CRASH_POINT";

fn vector(id: u64) -> Vector {
    let mut state = (id + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
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

fn open(path: &str) -> SrvDB {
    SrvDB::new_with_config(path, DatabaseConfig::new(DIM).unwrap()).unwrap()
}

fn add_range(db: &mut SrvDB, ids: std::ops::Range<u64>) {
    let vectors: Vec<Vector> = ids.clone().map(vector).collect();
    let metas: Vec<String> = ids.map(|i| format!(r#"{{"id": {i}}}"#)).collect();
    db.add_batch(&vectors, &metas).unwrap();
}

/// Child: commit COMMITTED vectors, then write 50 more and die at the
/// requested point of the next persist.
#[test]
fn crc_crash_child_entry() {
    let Ok(dir) = std::env::var(CHILD_DIR) else {
        return; // normal test run: nothing to do
    };
    let point = std::env::var(CHILD_POINT).unwrap();
    let dir = Path::new(&dir);
    let path = dir.to_str().unwrap();
    {
        let mut db = open(path);
        add_range(&mut db, 0..COMMITTED);
        db.persist().unwrap();
    }
    let side = dir.join("vectors.bin.crc");
    let old_sidecar = std::fs::read(&side).unwrap();

    // Step 1 of persist() for the vector file: write, fsync, new sidecar
    let mut storage = VectorStorage::new(path, DIM).unwrap();
    for id in COMMITTED..COMMITTED + 50 {
        storage.append(&vector(id).data).unwrap();
    }
    storage.flush().unwrap();

    if point == "fsync" {
        // Undo the rename: new table back in the temp file, old in place
        std::fs::rename(&side, dir.join(".vectors.bin.crc.tmp")).unwrap();
        std::fs::write(&side, old_sidecar).unwrap();
    }
    // The redb commit never happens
    std::process::abort();
}

fn crash_at(point: &str) {
    let dir = TempDir::new().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crc_crash_child_entry", "--nocapture"])
        .env(CHILD_DIR, dir.path())
        .env(CHILD_POINT, point)
        .status()
        .unwrap();
    assert!(!status.success(), "{point}: child should have aborted");

    let path = dir.path().to_str().unwrap();
    {
        let mut db = open(path);
        assert_eq!(db.count(), COMMITTED, "{point}");
        for id in [0, 1, 150, COMMITTED - 1] {
            let hit = &db.search(&vector(id), 1).unwrap()[0];
            assert_eq!(hit.id, id, "{point}");
            let meta = db.get_metadata(id).unwrap().unwrap();
            assert!(meta.contains(&format!("\"id\": {id}")), "{point}");
        }
        assert!(db.get_metadata(COMMITTED).unwrap().is_none(), "{point}");
        // Writing continues at the committed position
        add_range(&mut db, 1000..1010);
        db.persist().unwrap();
    }
    let db = open(path);
    assert_eq!(db.count(), COMMITTED + 10, "{point}");
    assert_eq!(db.search(&vector(1005), 1).unwrap()[0].id, COMMITTED + 5);
}

#[test]
fn crash_after_vector_fsync_before_sidecar_rename() {
    crash_at("fsync");
}

#[test]
fn crash_after_sidecar_before_redb_commit() {
    crash_at("sidecar");
}
