//! Crash test for graph links (ADR 0016): kill a writer that links,
//! deletes and compacts at random moments, reopen, verify that exactly
//! the links of the last successful persist() are there.
//!
//! The child adds originals in batches of BATCH (metadata `{"id": o}`),
//! links each original `o` to `o - 1` within its batch, deletes every
//! original divisible by 10, and persists (compacting every 4th round).
//! Kills: SRVDB_CRASH_ITERATIONS (default 5).

use srvdb::types::DatabaseConfig;
use srvdb::{SrvDB, Vector, VectorEngine};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

const DIM: usize = 128;
const BATCH: u64 = 40;
const CHILD_DIR: &str = "SRVDB_LINKS_CRASH_CHILD";

fn vector(o: u64) -> Vector {
    Vector::new(
        (0..DIM)
            .map(|j| ((o * 31 + j as u64 * 7) % 97) as f32)
            .collect(),
    )
}

fn open(dir: &Path) -> SrvDB {
    SrvDB::new_with_config(dir.to_str().unwrap(), DatabaseConfig::new(DIM).unwrap()).unwrap()
}

/// internal id -> original, for live records
fn originals(db: &SrvDB) -> HashMap<u64, u64> {
    db.all_metadata()
        .unwrap()
        .into_iter()
        .map(|(id, json)| {
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            (id, v["id"].as_u64().unwrap())
        })
        .collect()
}

fn next_original(db: &SrvDB) -> u64 {
    originals(db)
        .values()
        .max()
        .map_or(0, |&last| (last + 1).div_ceil(BATCH) * BATCH)
}

#[test]
fn links_crash_child_entry() {
    let Ok(dir) = std::env::var(CHILD_DIR) else {
        return; // normal test run
    };
    let mut db = open(Path::new(&dir));
    let mut next = next_original(&db);
    let mut stdout = std::io::stdout();
    let mut round = 0u64;
    loop {
        round += 1;
        let os = next..next + BATCH;
        let vectors: Vec<Vector> = os.clone().map(vector).collect();
        let metas: Vec<String> = os.clone().map(|o| format!(r#"{{"id": {o}}}"#)).collect();
        let ids = db.add_batch(&vectors, &metas).unwrap();
        for w in ids.windows(2) {
            db.link(w[1], "prev", w[0]).unwrap();
        }
        for (id, o) in ids.iter().zip(os) {
            if o % 10 == 0 {
                db.delete(*id).unwrap();
            }
        }
        next += BATCH;
        if round.is_multiple_of(4) {
            db.compact().unwrap();
        } else {
            db.persist().unwrap();
        }
        writeln!(stdout, "COMMITTED {next}").unwrap();
        stdout.flush().unwrap();
    }
}

fn run_and_kill(dir: &Path) -> u64 {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "links_crash_child_entry",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_DIR, dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let (mut last, mut line) = (0, Vec::new());
        while reader.read_until(b'\n', &mut line).unwrap_or(0) > 0 {
            if let Some(complete) = line.strip_suffix(b"\n") {
                let text = String::from_utf8_lossy(complete);
                if let Some(Ok(n)) = text.rsplit_once("COMMITTED ").map(|(_, n)| n.parse()) {
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

fn verify(dir: &Path, reported: u64, previous: u64) -> u64 {
    let db = open(dir);
    let floor = reported.max(previous);
    let next = next_original(&db);
    assert!(next == floor || next == floor + BATCH, "{next} vs {floor}");
    let by_id = originals(&db);
    let id_of: HashMap<u64, u64> = by_id.iter().map(|(&id, &o)| (o, id)).collect();
    let expected_live: Vec<u64> = (0..next).filter(|o| o % 10 != 0).collect();
    assert_eq!(id_of.len(), expected_live.len(), "live records");
    let mut expected_links = 0;
    for o in expected_live {
        let want = if o % BATCH != 0 && (o - 1) % 10 != 0 {
            expected_links += 1;
            vec![("prev".to_string(), id_of[&(o - 1)])]
        } else {
            vec![]
        };
        assert_eq!(db.neighbors(id_of[&o], None).unwrap(), want, "links of {o}");
    }
    assert_eq!(db.link_count(), expected_links, "no stale or extra links");
    next
}

#[test]
fn crash_links() {
    let iterations: usize = std::env::var("SRVDB_CRASH_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let dir = tempfile::TempDir::new().unwrap();
    let mut committed = 0;
    for _ in 0..iterations {
        let reported = run_and_kill(dir.path());
        committed = verify(dir.path(), reported, committed);
    }
    eprintln!("links: {iterations} kills, {committed} originals verified");
}
