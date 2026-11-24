//! Crash test for the memory-budget step-down (ADR 0018), following
//! `test_crash_settings.rs`: a child process grows a budgeted database
//! through flat and RaBitQ 4, 2 and 1 bits and is killed at random
//! moments; the parent reopens with only the path every time.
//! Kills: SRVDB_CRASH_ITERATIONS (default 5).

use srvdb::{CreateOptions, IndexChoice, SrvDB, Vector, VectorEngine};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::Duration;
use tempfile::TempDir;

const DIM: usize = 32;
const BATCH: u64 = 1500;
const CHILD_DIR: &str = "SRVDB_BUDGET_CRASH_CHILD";

fn vector(i: u64) -> Vector {
    let mut s = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
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
}

/// Child: add batches and persist until killed; once the budget refuses,
/// keep persisting
#[test]
fn budget_crash_child_entry() {
    let Ok(dir) = std::env::var(CHILD_DIR) else {
        return; // normal test run
    };
    let mut db = SrvDB::open_existing(&dir).unwrap();
    loop {
        let from = db.count();
        let vectors: Vec<Vector> = (from..from + BATCH).map(vector).collect();
        match db.add_batch(&vectors, &vec!["{}".to_string(); BATCH as usize]) {
            Ok(_) => {}
            Err(srvdb::Error::MemoryBudget(_)) => {}
            Err(e) => panic!("{e}"),
        }
        db.persist().unwrap();
        println!("PERSISTED {}", db.count());
    }
}

#[test]
fn killed_while_stepping_down_reopens_within_budget() {
    let iterations: usize = std::env::var("SRVDB_CRASH_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let root = TempDir::new().unwrap();
    let mut persisted = 0;
    let mut stepped = false;
    for i in 0..iterations {
        // A fresh database each time, so the kill can land on any rung
        let dir = root.path().join(format!("db{i}"));
        let path = dir.to_str().unwrap();
        let mut options = CreateOptions::new(IndexChoice::Auto);
        options.set("memory_mb", 1).unwrap();
        drop(SrvDB::create(path, DIM, &options).unwrap());

        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "budget_crash_child_entry", "--nocapture"])
            .args(["--test-threads=1"])
            .env(CHILD_DIR, path)
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
        std::thread::sleep(Duration::from_millis(50 + rand::random::<u64>() % 1500));
        child.kill().unwrap();
        child.wait().unwrap();
        persisted += reader.join().unwrap();

        let db = SrvDB::open_existing(path).unwrap();
        let report = db.memory_report();
        // Whole batches only, and the recorded mode holds them in budget
        assert_eq!(db.count() % BATCH, 0);
        assert_eq!(report.budget_bytes, Some(1 << 20));
        assert!(
            report.total_bytes <= report.limit_bytes.unwrap(),
            "{report:?}"
        );
        assert!(matches!(report.mode, "flat" | "rabitq"), "{report:?}");
        stepped |= report.mode == "rabitq";
        if db.count() > 0 {
            let probe = db.count() - 1;
            assert_eq!(db.search(&vector(probe), 1).unwrap()[0].id, probe);
        }
    }
    assert!(persisted > 0, "the child never persisted; raise the delay");
    assert!(stepped, "no kill came after a step-down; raise the delay");
}
