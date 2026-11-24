//! Memory budget (ADR 0018): selection at the boundaries, step-down,
//! refusal, reopen and option validation

use srvdb::{CreateOptions, Error, IndexChoice, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 32;

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

/// Add records until the database holds `to`
fn fill(db: &mut SrvDB, to: u64) {
    let (from, n) = (db.count(), to - db.count());
    let vectors: Vec<Vector> = (from..from + n).map(vector).collect();
    db.add_batch(&vectors, &vec!["{}".to_string(); n as usize])
        .unwrap();
}

fn budgeted(path: &str, memory_mb: u64) -> SrvDB {
    let mut options = CreateOptions::new(IndexChoice::Auto);
    options.set("memory_mb", memory_mb).unwrap();
    SrvDB::create(path, DIM, &options).unwrap()
}

fn mode(db: &SrvDB) -> (&'static str, Option<u8>) {
    let r = db.memory_report();
    (r.mode, r.bits)
}

#[test]
fn steps_down_exactly_past_each_capacity_and_then_refuses() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");
    let path = path.to_str().unwrap();
    let mut db = budgeted(path, 1);
    let report = db.memory_report();
    assert_eq!(mode(&db), ("flat", None));
    assert!(report.exact);
    assert_eq!(report.budget_bytes, Some(1 << 20));
    assert_eq!(report.limit_bytes, Some((1 << 20) / 100 * 80));

    let expected: [(&str, Option<u8>); 3] = [
        ("rabitq", Some(4)),
        ("rabitq", Some(2)),
        ("rabitq", Some(1)),
    ];
    for next in expected {
        let report = db.memory_report();
        let cap = report.capacity.unwrap();
        // Filling to capacity keeps the mode
        fill(&mut db, cap);
        assert_eq!(db.memory_report().mode, report.mode);
        assert_eq!(db.memory_report().bits, report.bits);
        assert!(db.memory_report().total_bytes <= report.limit_bytes.unwrap());
        // One more record steps down
        fill(&mut db, cap + 1);
        assert_eq!(mode(&db), next);
        let after = db.memory_report();
        assert!(!after.exact);
        assert!(after.total_bytes <= after.limit_bytes.unwrap());
        // Every vector is still found after re-encoding
        let probe = db.count() / 2;
        assert_eq!(db.search(&vector(probe), 1).unwrap()[0].id, probe);
    }

    // At 1-bit capacity nothing fits any more: refused, nothing written
    let cap = db.memory_report().capacity.unwrap();
    fill(&mut db, cap);
    let before = db.count();
    let err = db
        .add_batch(&[vector(before)], &["{}".to_string()])
        .unwrap_err();
    assert!(matches!(err, Error::MemoryBudget(_)), "{err:?}");
    assert!(err.to_string().contains("memory_mb"), "{err}");
    assert_eq!(db.count(), before);
    assert!(matches!(
        db.add(&vector(before), "{}").unwrap_err(),
        Error::MemoryBudget(_)
    ));
    db.persist().unwrap();
    drop(db);

    // Reopen with only the path: same mode, same budget, same data
    let db = SrvDB::open_existing(path).unwrap();
    assert_eq!(mode(&db), ("rabitq", Some(1)));
    assert_eq!(db.count(), before);
    assert_eq!(db.memory_report().budget_bytes, Some(1 << 20));
    assert_eq!(db.search(&vector(7), 1).unwrap()[0].id, 7);
}

#[test]
fn a_batch_that_skips_rungs_lands_on_the_best_fit() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");
    let mut db = budgeted(path.to_str().unwrap(), 1);
    // Far past flat and 4-bit capacity, within 2 bits
    fill(&mut db, 40_000);
    assert_eq!(mode(&db), ("rabitq", Some(2)));
}

#[test]
fn reopen_keeps_flat_and_budget_and_exist_ok_matches_it() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");
    let path = path.to_str().unwrap();
    let mut db = budgeted(path, 3);
    fill(&mut db, 100);
    db.persist().unwrap();
    drop(db);

    let mut options = CreateOptions::new(IndexChoice::Auto);
    options.set("memory_mb", 3).unwrap();
    options.set_exist_ok(true);
    let db = SrvDB::create(path, DIM, &options).unwrap();
    assert_eq!(db.index_name(), "flat");
    assert_eq!(db.memory_report().budget_bytes, Some(3 << 20));
    drop(db);

    options.set("memory_mb", 4).unwrap();
    let Err(err) = SrvDB::create(path, DIM, &options) else {
        panic!("another budget was accepted");
    };
    assert!(matches!(err, Error::SettingsMismatch(_)), "{err:?}");
}

#[test]
fn memory_mb_is_validated_and_only_for_auto() {
    let dir = TempDir::new().unwrap();
    for bad in [0, (1 << 20) + 1] {
        let mut options = CreateOptions::new(IndexChoice::Auto);
        options.set("memory_mb", bad).unwrap();
        let path = dir.path().join(format!("bad{bad}"));
        let Err(err) = SrvDB::create(path.to_str().unwrap(), DIM, &options) else {
            panic!("memory_mb = {bad} was accepted");
        };
        assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
        assert!(!path.exists());
    }
    let err = CreateOptions::new(IndexChoice::Hnsw)
        .set("memory_mb", 50)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}

#[test]
fn every_mode_reports_its_parts() {
    let dir = TempDir::new().unwrap();
    let flat = dir.path().join("flat");
    let mut db = SrvDB::create(
        flat.to_str().unwrap(),
        DIM,
        &CreateOptions::new(IndexChoice::Flat),
    )
    .unwrap();
    fill(&mut db, 10);
    let r = db.memory_report();
    assert_eq!((r.vector_bytes, r.index_bytes), (10 * 4 * DIM as u64, 0));
    assert_eq!(r.budget_bytes, None);
    assert_eq!(r.capacity, None);

    let hnsw = dir.path().join("hnsw");
    let mut db = SrvDB::create(
        hnsw.to_str().unwrap(),
        DIM,
        &CreateOptions::new(IndexChoice::Hnsw),
    )
    .unwrap();
    fill(&mut db, 10);
    let r = db.memory_report();
    assert_eq!(r.index_bytes, 10 * (16 * 16 + 96));
    assert!(!r.exact);
    assert_eq!(
        r.total_bytes,
        r.vector_bytes + r.index_bytes + r.links_bytes + r.keyword_bytes
    );
}
