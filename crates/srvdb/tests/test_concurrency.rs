//! Concurrency contract (ADR 0005): one open instance per database;
//! searches run in parallel through &SrvDB; writes need &mut SrvDB.

use srvdb::types::DatabaseConfig;
use srvdb::{Error, SrvDB, Vector, VectorEngine};
use std::sync::{Arc, RwLock};
use std::thread;
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

fn open(dir: &TempDir) -> srvdb::Result<SrvDB> {
    SrvDB::new_with_config(
        dir.path().to_str().unwrap(),
        DatabaseConfig::new(DIM).unwrap(),
    )
}

fn filled(dir: &TempDir, n: u64) -> SrvDB {
    let mut db = open(dir).unwrap();
    let vectors: Vec<Vector> = (0..n).map(vector).collect();
    let metas: Vec<String> = (0..n).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    db.add_batch(&vectors, &metas).unwrap();
    db
}

#[test]
fn srvdb_can_be_shared_between_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SrvDB>();
}

#[test]
fn a_second_open_of_the_same_database_is_refused() {
    let dir = TempDir::new().unwrap();
    let mut first = filled(&dir, 10);
    first.persist().unwrap();
    let err = open(&dir).err().unwrap();
    assert!(matches!(err, Error::Locked(_)), "{err:?}");
    assert!(err.to_string().contains("already open"));

    drop(first);
    assert_eq!(open(&dir).unwrap().count(), 10);
}

#[test]
fn searches_run_in_parallel_through_a_shared_reference() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(filled(&dir, 500));
    let handles: Vec<_> = (0..8u64)
        .map(|t| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                for i in (t..500).step_by(8) {
                    assert_eq!(db.search(&vector(i), 1).unwrap()[0].id, i);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn a_writer_and_readers_share_the_database_through_a_rwlock() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(RwLock::new(filled(&dir, 100)));

    let writer = {
        let db = Arc::clone(&db);
        thread::spawn(move || {
            for i in 100..300 {
                let mut db = db.write().unwrap();
                db.add(&vector(i), &format!(r#"{{"i": {i}}}"#)).unwrap();
            }
        })
    };
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                for i in 0..100 {
                    let db = db.read().unwrap();
                    assert_eq!(db.search(&vector(i), 1).unwrap()[0].id, i);
                }
            })
        })
        .collect();
    writer.join().unwrap();
    for reader in readers {
        reader.join().unwrap();
    }
    assert_eq!(db.read().unwrap().count(), 300);
}
