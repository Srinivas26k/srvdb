//! Crash test (ADR 0003): kill a writer at random moments, reopen, verify.
//!
//! The test binary re-runs itself as a child process (selected by the
//! SRVDB_CRASH_CHILD variable) that loops: add a batch, persist(), print
//! "COMMITTED <count>". The parent kills it after a random delay (SIGKILL
//! on Unix), reopens the database and checks that exactly the committed
//! data is there and intact. Repeated on the same database, so recovery
//! itself is crashed into as well.
//!
//! Kills per mode: SRVDB_CRASH_ITERATIONS (default 5; CI runs 1000 in
//! total). This exercises process crashes; power loss additionally relies
//! on the fsync ordering in the commit protocol.

use srvdb::types::{DatabaseConfig, IndexType, QuantizationMode};
use srvdb::{SrvDB, Vector, VectorEngine};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

const DIM: usize = 128;
const BATCH: u64 = 50;
const CHILD_DIR: &str = "SRVDB_CRASH_CHILD";
const CHILD_MODE: &str = "SRVDB_CRASH_MODE";

/// Deterministic vector for an id, so the parent can verify contents
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

fn meta(id: u64) -> String {
    format!(r#"{{"id": {id}}}"#)
}

fn open(dir: &Path, mode: &str) -> SrvDB {
    let mut config = DatabaseConfig::new(DIM).unwrap();
    match mode {
        "hnsw" => config.index_type = IndexType::HNSW,
        "sq8" => {
            config.quantization.enabled = true;
            config.quantization.mode = QuantizationMode::Scalar;
        }
        _ => {}
    }
    SrvDB::new_with_config(dir.to_str().unwrap(), config).unwrap()
}

/// Child mode: write and commit batches until killed.
#[test]
fn crash_child_entry() {
    let Ok(dir) = std::env::var(CHILD_DIR) else {
        return; // normal test run: nothing to do
    };
    let mode = std::env::var(CHILD_MODE).unwrap();
    let mut db = open(Path::new(&dir), &mode);
    if mode == "churn" {
        churn_child(db);
    }
    let mut stdout = std::io::stdout();
    loop {
        let first = db.count();
        let ids = first..first + BATCH;
        let vectors: Vec<Vector> = ids.clone().map(vector).collect();
        let metas: Vec<String> = ids.map(meta).collect();
        db.add_batch(&vectors, &metas).unwrap();
        db.persist().unwrap();
        writeln!(stdout, "COMMITTED {}", db.count()).unwrap();
        stdout.flush().unwrap();
    }
}

/// Churn child: each round adds a batch, deletes every original number
/// divisible by 7, and compacts every 4th round (otherwise persists).
/// Reports the number of originals added so far.
fn churn_child(mut db: SrvDB) -> ! {
    let mut next = next_original(&db);
    let mut stdout = std::io::stdout();
    for round in 1.. {
        let originals = next..next + BATCH;
        let vectors: Vec<Vector> = originals.clone().map(vector).collect();
        let metas: Vec<String> = originals.clone().map(meta).collect();
        let ids = db.add_batch(&vectors, &metas).unwrap();
        for (id, original) in ids.into_iter().zip(originals) {
            if original % 7 == 0 {
                db.delete(id).unwrap();
            }
        }
        next += BATCH;
        if round % 4 == 0 {
            db.compact().unwrap(); // persists first
        } else {
            db.persist().unwrap();
        }
        writeln!(stdout, "COMMITTED {next}").unwrap();
        stdout.flush().unwrap();
    }
    unreachable!()
}

/// (internal id, original number) of every live vector
fn originals(db: &SrvDB) -> Vec<(u64, u64)> {
    let mut found: Vec<(u64, u64)> = db
        .all_metadata()
        .unwrap()
        .into_iter()
        .map(|(id, json)| {
            let value: serde_json::Value = serde_json::from_str(&json).unwrap();
            (id, value["id"].as_u64().unwrap())
        })
        .collect();
    found.sort_by_key(|&(_, original)| original);
    found
}

/// Originals are added in whole batches, and every batch keeps some
fn next_original(db: &SrvDB) -> u64 {
    originals(db)
        .last()
        .map_or(0, |&(_, last)| (last + 1).div_ceil(BATCH) * BATCH)
}

fn verify_churn(dir: &Path, last_reported: u64, previous: u64) -> u64 {
    let db = open(dir, "churn");
    let floor = last_reported.max(previous);
    let next = next_original(&db);
    assert!(
        next == floor || next == floor + BATCH,
        "churn: {next} originals added, last confirmed {floor}"
    );
    // Exactly the committed adds minus the committed deletes survive
    let found = originals(&db);
    let survivors: Vec<u64> = found.iter().map(|&(_, o)| o).collect();
    let expected: Vec<u64> = (0..next).filter(|o| o % 7 != 0).collect();
    assert_eq!(survivors, expected, "churn: surviving originals");
    assert_eq!(db.live_count(), expected.len() as u64, "churn: live count");
    for &(id, original) in found.iter().step_by((found.len() / 30).max(1)) {
        let hit = db.search(&vector(original), 1).unwrap();
        assert_eq!(hit[0].id, id, "churn: original {original} not at its id");
    }
    next
}

/// Run the child until a random moment, kill it, return the last count it
/// reported as committed.
fn run_and_kill(dir: &Path, mode: &str) -> u64 {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "crash_child_entry",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_DIR, dir)
        .env(CHILD_MODE, mode)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        // Only complete lines count (a kill can cut "COMMITTED 800" short),
        // and the first report shares a line with libtest's "test ... "
        let mut reader = BufReader::new(stdout);
        let mut last = 0;
        let mut line = Vec::new();
        while reader.read_until(b'\n', &mut line).unwrap_or(0) > 0 {
            if let Some(complete) = line.strip_suffix(b"\n") {
                let text = String::from_utf8_lossy(complete);
                if let Some(Ok(n)) = text
                    .rsplit_once("COMMITTED ")
                    .map(|(_, n)| n.parse::<u64>())
                {
                    last = n;
                }
            }
            line.clear();
        }
        last
    });
    std::thread::sleep(Duration::from_millis(5 + rand::random::<u64>() % 145));
    child.kill().unwrap();
    child.wait().unwrap();
    reader.join().unwrap()
}

fn verify(dir: &Path, mode: &str, last_reported: u64, previous: u64) -> u64 {
    let db = open(dir, mode);
    let count = db.count();

    // Committed data is never lost: at least the last confirmed commit,
    // at most one more batch (killed after commit, before printing)
    let floor = last_reported.max(previous);
    assert!(
        count == floor || count == floor + BATCH,
        "{mode}: count {count}, last confirmed {floor}"
    );
    assert_eq!(count % BATCH, 0, "{mode}: partial batch survived");

    // Metadata matches the vectors exactly
    let metadata = db.all_metadata().unwrap();
    assert_eq!(metadata.len() as u64, count, "{mode}: metadata count");
    for (id, json) in &metadata {
        assert_eq!(json, &meta(*id), "{mode}: metadata of {id}");
    }

    // Stored vectors are intact and at their own ids (sample for speed)
    let step = (count / 40).max(1);
    for id in (0..count)
        .step_by(step as usize)
        .chain(count.saturating_sub(3)..count)
    {
        let hit = db.search(&vector(id), 1).unwrap();
        assert_eq!(hit[0].id, id, "{mode}: vector {id} not found at its id");
    }

    // The vector file holds exactly the committed records
    if mode != "sq8" {
        let len = std::fs::metadata(dir.join("vectors.bin")).unwrap().len();
        assert_eq!(
            len,
            24 + count * (DIM as u64) * 4,
            "{mode}: vectors.bin size"
        );
    }
    count
}

fn crash_repeatedly(mode: &str) {
    let iterations: usize = std::env::var("SRVDB_CRASH_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    // A fresh database every KILLS_PER_DATABASE kills keeps each kill's cost
    // roughly constant (the HNSW graph is rewritten on every persist).
    const KILLS_PER_DATABASE: usize = 25;
    let mut verified = 0;
    for start in (0..iterations).step_by(KILLS_PER_DATABASE) {
        let dir = tempfile::TempDir::new().unwrap();
        let mut committed = 0;
        for _ in start..iterations.min(start + KILLS_PER_DATABASE) {
            let reported = run_and_kill(dir.path(), mode);
            committed = if mode == "churn" {
                verify_churn(dir.path(), reported, committed)
            } else {
                verify(dir.path(), mode, reported, committed)
            };
        }
        verified += committed;
    }
    eprintln!("{mode}: {iterations} kills, {verified} committed vectors verified");
}

#[test]
fn crash_flat() {
    crash_repeatedly("flat");
}

#[test]
fn crash_hnsw() {
    crash_repeatedly("hnsw");
}

#[test]
fn crash_sq8() {
    crash_repeatedly("sq8");
}

#[test]
fn crash_churn() {
    crash_repeatedly("churn");
}
