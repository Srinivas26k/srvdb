//! A memory budget survives every constructor (ADR 0018): reopening with
//! `new_rabitq` or `new_with_config` instead of `open_existing` must keep
//! the budget enforced and `index_settings.bin` at payload format 2.

use srvdb::{
    CreateOptions, DatabaseConfig, Error, IndexChoice, IndexType, QuantizationConfig, RabitqConfig,
    SrvDB, Vector, VectorEngine,
};
use std::path::Path;
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

fn add(db: &mut SrvDB, n: u64) -> srvdb::Result<Vec<u64>> {
    let from = db.count();
    let vectors: Vec<Vector> = (from..from + n).map(vector).collect();
    db.add_batch(&vectors, &vec!["{}".to_string(); n as usize])
}

/// Payload format of `index_settings.bin`: the first payload byte after
/// the 24-byte envelope header (a postcard varint, 1 or 2 here)
fn settings_format(dir: &Path) -> u8 {
    std::fs::read(dir.join("index_settings.bin")).unwrap()[24]
}

fn budgeted(dir: &Path) {
    let mut options = CreateOptions::new(IndexChoice::Auto);
    options.set("memory_mb", 1).unwrap();
    drop(SrvDB::create(dir.to_str().unwrap(), DIM, &options).unwrap());
    assert_eq!(settings_format(dir), 2);
}

/// The budget is in force: the database reports it and refuses an add
/// far past 1-bit capacity
fn assert_enforced(db: &mut SrvDB) {
    assert_eq!(db.memory_report().budget_bytes, Some(1 << 20));
    let before = db.count();
    let err = add(db, 200_000).unwrap_err();
    assert!(matches!(err, Error::MemoryBudget(_)), "{err:?}");
    assert_eq!(db.count(), before);
}

#[test]
fn new_rabitq_keeps_the_budget_and_format_2() {
    let root = TempDir::new().unwrap();
    let dir = root.path().join("db");
    budgeted(&dir);
    let path = dir.to_str().unwrap();
    let mut db = SrvDB::open_existing(path).unwrap();
    let cap = db.memory_report().capacity.unwrap();
    add(&mut db, cap + 1).unwrap();
    assert_eq!(db.memory_report().bits, Some(4));
    db.persist().unwrap();
    drop(db);

    let mut db = SrvDB::new_rabitq(path, DIM, RabitqConfig::new(4).unwrap()).unwrap();
    add(&mut db, 10).unwrap();
    db.persist().unwrap();
    assert_eq!(settings_format(&dir), 2);
    assert_enforced(&mut db);
}

#[test]
fn new_with_config_keeps_the_budget_and_format_2() {
    let root = TempDir::new().unwrap();
    let dir = root.path().join("db");
    budgeted(&dir);
    let config = DatabaseConfig {
        dimension: DIM,
        quantization: QuantizationConfig::default(),
        index_type: IndexType::Flat,
    };
    let mut db = SrvDB::new_with_config(dir.to_str().unwrap(), config).unwrap();
    add(&mut db, 10).unwrap();
    db.persist().unwrap();
    assert_eq!(settings_format(&dir), 2);
    assert_enforced(&mut db);
}
