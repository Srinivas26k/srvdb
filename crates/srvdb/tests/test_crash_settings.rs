//! Crash test for `index_settings.bin` (ADR 0011), following the pattern
//! of `test_crash.rs`: a child process rewrites the settings in a loop and
//! is killed at random moments; the parent reopens with only the path.
//! Kills: SRVDB_CRASH_ITERATIONS (default 5).

use srvdb::{HNSWConfig, IndexMode, SrvDB, Vector, VectorEngine};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;
use tempfile::TempDir;

const DIM: usize = 128;
const CHILD_DIR: &str = "SRVDB_SETTINGS_CRASH_CHILD";

fn vector(i: u64) -> Vector {
    Vector::new(
        (0..DIM)
            .map(|j| ((i as usize * 7 + j * 13) % 29) as f32 - 14.0 + i as f32 * 0.01)
            .collect(),
    )
}

fn hnsw_db(dir: &Path) {
    let path = dir.to_str().unwrap();
    let mut db = SrvDB::new_with_hnsw(path, DIM, HNSWConfig::new(8, 64, 40)).unwrap();
    let vectors: Vec<Vector> = (0..50).map(vector).collect();
    db.add_batch(&vectors, &vec!["{}".to_string(); 50]).unwrap();
    db.persist().unwrap();
}

/// Child: reopen with the stored settings, switch ef_search (which rewrites
/// index_settings.bin on every persist) until killed.
#[test]
fn settings_crash_child_entry() {
    let Ok(dir) = std::env::var(CHILD_DIR) else {
        return; // normal test run
    };
    let mut db = SrvDB::open_existing(&dir).unwrap();
    let mut ef = 20;
    loop {
        ef = if ef == 20 { 200 } else { 20 };
        db.set_ef_search(ef).unwrap();
        db.persist().unwrap();
        println!("PERSISTED {ef}");
    }
}

/// Kill the writer of index_settings.bin at random moments: the database
/// always reopens in HNSW mode with an ef_search it committed, and no
/// half-written settings file is ever read.
#[test]
fn killed_while_writing_settings_reopens_with_old_or_new_settings() {
    let dir = TempDir::new().unwrap();
    hnsw_db(dir.path());
    let iterations: usize = std::env::var("SRVDB_CRASH_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let mut persisted = 0;
    for _ in 0..iterations {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "settings_crash_child_entry", "--nocapture"])
            .args(["--test-threads=1"])
            .env(CHILD_DIR, dir.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            BufReader::new(stdout)
                .lines()
                .map_while(|l| l.ok())
                .filter(|l| l.contains("PERSISTED"))
                .count()
        });
        std::thread::sleep(Duration::from_millis(20 + rand::random::<u64>() % 130));
        child.kill().unwrap();
        child.wait().unwrap();
        persisted += reader.join().unwrap();

        let db = SrvDB::open_existing(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(db.current_mode, IndexMode::Hnsw);
        assert_eq!(db.count(), 50);
        assert_eq!(db.search(&vector(7), 1).unwrap()[0].id, 7);
    }
    // The kills landed while settings were being rewritten
    assert!(persisted > 0, "the child never persisted; raise the delay");
}
