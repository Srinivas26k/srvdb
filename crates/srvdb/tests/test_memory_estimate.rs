//! The memory estimates of ADR 0018 against measured heap allocations.
//!
//! A counting global allocator tracks live heap bytes. Every measurement
//! is a difference between two database sizes, so fixed costs (redb,
//! metadata, the test harness) cancel. One test function only, so no other
//! test allocates concurrently in this binary.

use srvdb::{CreateOptions, HNSWConfig, IndexChoice, SrvDB, Vector, VectorEngine};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use tempfile::TempDir;

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);
static NEXT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to the system allocator; the
// counter only observes sizes.
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
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        LIVE.fetch_add(new as isize - layout.size() as isize, Ordering::Relaxed);
        // SAFETY: same contract as the caller's
        unsafe { System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn live() -> isize {
    LIVE.load(Ordering::Relaxed)
}

fn vectors(n: usize, dim: usize) -> Vec<Vector> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            Vector::new(
                (0..dim)
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

fn fill(db: &mut SrvDB, data: &[Vector]) {
    db.add_batch(data, &vec!["{}".to_string(); data.len()])
        .unwrap();
    db.persist().unwrap();
}

/// Heap bytes an open database of `index` holds, and its estimate
fn held(dir: &TempDir, data: &[Vector], index: IndexChoice, bits: Option<u64>) -> (i64, i64) {
    let dim = data[0].data.len();
    let name = format!("{index}-{}", NEXT.fetch_add(1, Ordering::Relaxed));
    let path = dir.path().join(name);
    let path = path.to_str().unwrap();
    let mut options = CreateOptions::new(index);
    if let Some(bits) = bits {
        options.set("bits", bits).unwrap();
    }
    let mut db = SrvDB::create(path, dim, &options).unwrap();
    fill(&mut db, data);
    drop(db);
    let before = live();
    let db = SrvDB::open_existing(path).unwrap();
    let held = (live() - before) as i64;
    let r = db.memory_report();
    (held, (r.vector_bytes + r.index_bytes) as i64)
}

/// RaBitQ heap less what a flat database of the same vectors holds
/// (redb's cache, metadata), and the RaBitQ estimate
fn rabitq_open(dir: &TempDir, n: usize, dim: usize, bits: u8) -> (i64, i64) {
    let data = vectors(n, dim);
    let (flat, _) = held(dir, &data, IndexChoice::Flat, None);
    let (codes, estimate) = held(dir, &data, IndexChoice::Rabitq, Some(u64::from(bits)));
    (codes - flat, estimate)
}

fn hnsw_built(dir: &TempDir, n: usize, dim: usize) -> (i64, i64) {
    let path = dir.path().join(format!("hnsw-{n}"));
    let before = live();
    let mut db =
        SrvDB::new_with_hnsw(path.to_str().unwrap(), dim, HNSWConfig::new(16, 64, 32)).unwrap();
    let data = vectors(n, dim);
    let data_bytes = live() - before;
    fill(&mut db, &data);
    drop(data);
    let held = (live() - before - data_bytes) as i64;
    (held, db.memory_report().index_bytes as i64)
}

fn within(measured: i64, estimated: i64, tolerance: f64, what: &str) {
    let error = (measured - estimated).abs() as f64 / estimated as f64;
    eprintln!("{what}: measured {measured}, estimated {estimated}");
    assert!(
        error <= tolerance,
        "{what}: measured {measured} bytes, estimated {estimated} ({:.1} %)",
        error * 100.0
    );
}

#[test]
fn estimates_match_measured_heap() {
    let dir = TempDir::new().unwrap();
    let _ = SrvDB::create(
        dir.path().join("warm").to_str().unwrap(),
        8,
        &CreateOptions::new(IndexChoice::Flat),
    )
    .map(drop); // first-use allocations (rayon pool, loggers) happen here

    // RaBitQ: the scan copy per vector, at each bit width (within 2 %)
    for bits in [1u8, 2, 4] {
        let (small, est_small) = rabitq_open(&dir, 1_000, 128, bits);
        let (large, est_large) = rabitq_open(&dir, 9_000, 128, bits);
        within(
            large - small,
            est_large - est_small,
            0.02,
            &format!("{bits}-bit codes"),
        );
    }
    // RaBitQ: the rotation, from two dimensions at the same count
    let (d64, e64) = rabitq_open(&dir, 64, 64, 2);
    let (d512, e512) = rabitq_open(&dir, 64, 512, 2);
    within(d512 - d64, e512 - e64, 0.02, "rotation");

    // HNSW graph: within 35 % (neighbour-list capacity depends on order)
    let (small, est_small) = hnsw_built(&dir, 500, 16);
    let (large, est_large) = hnsw_built(&dir, 2_500, 16);
    within(large - small, est_large - est_small, 0.35, "hnsw graph");
}
