//! Memory per link (ADR 0016), measured with a counting allocator: live
//! heap bytes after linking and persisting, divided by the link count.
//! Includes whatever redb keeps allocated after the commit, so it is an
//! upper bound for the in-memory graph.

use srvdb::types::DatabaseConfig;
use srvdb::{SrvDB, Vector, VectorEngine};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);

// SAFETY: forwards every call to the system allocator unchanged and only
// records the size of each allocation and deallocation.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: same contract as the caller's
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: same contract as the caller's
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[test]
fn memory_per_link_is_bounded() {
    const RECORDS: u64 = 10_000;
    const FAN_OUT: u64 = 10;
    let dir = tempfile::TempDir::new().unwrap();
    let config = DatabaseConfig::new(128).unwrap();
    let mut db = SrvDB::new_with_config(dir.path().to_str().unwrap(), config).unwrap();
    let vectors: Vec<Vector> = (0..RECORDS)
        .map(|i| Vector::new(vec![i as f32 + 1.0; 128]))
        .collect();
    db.add_batch(&vectors, &vec!["{}".to_string(); RECORDS as usize])
        .unwrap();
    drop(vectors);
    db.persist().unwrap();

    let before = LIVE.load(Ordering::Relaxed);
    for src in 0..RECORDS {
        for k in 1..=FAN_OUT {
            let ty = if k % 2 == 0 { "cites" } else { "part_of" };
            db.link(src, ty, (src + k * 97) % RECORDS).unwrap();
        }
    }
    db.persist().unwrap();
    let grown = LIVE.load(Ordering::Relaxed) - before;
    let links = db.link_count();
    assert_eq!(links, RECORDS * FAN_OUT);
    let per_link = grown as f64 / links as f64;
    eprintln!("links: {links}, heap growth {grown} bytes, {per_link:.1} bytes per link");
    assert!(per_link < 128.0, "{per_link:.1} bytes per link");
}
