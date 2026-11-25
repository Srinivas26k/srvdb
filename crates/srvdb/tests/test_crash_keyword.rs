//! Crash test for `keyword.index` (ADR 0015), following `test_crash.rs`:
//! a child adds, deletes and persists in a loop with keyword search on and
//! is killed at random moments; the parent reopens and checks the keyword
//! index against the committed metadata.
//! Kills: SRVDB_CRASH_ITERATIONS (default 5).

use srvdb::{SrvDB, Vector, VectorEngine};
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::Duration;
use tempfile::TempDir;

const DIM: usize = 128;
const CHILD_DIR: &str = "SRVDB_KEYWORD_CRASH_CHILD";

fn config() -> srvdb::DatabaseConfig {
    srvdb::DatabaseConfig::new(DIM).unwrap()
}

fn vector(i: u64) -> Vector {
    Vector::new(
        (0..DIM)
            .map(|j| ((i as usize + j) % 13) as f32 + 1.0)
            .collect(),
    )
}

/// Child: add a record whose text names its id, delete every third, persist
#[test]
fn keyword_crash_child_entry() {
    let Ok(dir) = std::env::var(CHILD_DIR) else {
        return; // normal test run
    };
    let mut db = SrvDB::new_with_config(&dir, config()).unwrap();
    db.enable_keyword_index("text").unwrap();
    loop {
        let n = db.count();
        let meta = serde_json::json!({ "text": format!("common w{n}") }).to_string();
        let id = db.add(&vector(n), &meta).unwrap();
        if id.is_multiple_of(3) {
            db.delete(id).unwrap();
        }
        db.persist().unwrap();
        println!("PERSISTED {id}");
    }
}

/// After every kill, keyword search finds exactly the live committed
/// records: no lost, deleted or uncommitted record is returned.
#[test]
fn killed_while_persisting_keyword_index_reopens_consistent() {
    let dir = TempDir::new().unwrap();
    let iterations: usize = std::env::var("SRVDB_CRASH_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    for _ in 0..iterations {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "keyword_crash_child_entry", "--nocapture"])
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
        std::thread::sleep(Duration::from_millis(30 + rand::random::<u64>() % 150));
        child.kill().unwrap();
        child.wait().unwrap();
        let persisted = reader.join().unwrap() as u64;

        let db = SrvDB::new_with_config(dir.path().to_str().unwrap(), config()).unwrap();
        assert!(db.count() >= persisted);
        let live: HashSet<u64> = db
            .all_metadata()
            .unwrap()
            .into_iter()
            .map(|m| m.0)
            .collect();
        if db.keyword_field().unwrap().is_none() {
            // Killed before the first persist wrote the index
            continue;
        }
        let found: HashSet<u64> = db
            .keyword_search("common", 1_000_000, None)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(found, live);
        for id in &live {
            let hit = db.keyword_search(&format!("w{id}"), 2, None).unwrap();
            assert_eq!(hit.iter().map(|r| r.id).collect::<Vec<_>>(), [*id]);
        }
    }
}
