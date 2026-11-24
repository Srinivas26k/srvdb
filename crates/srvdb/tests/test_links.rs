//! Graph links (ADR 0016): durability, delete cascade, replace, compaction,
//! bounds, expansion and migration from a database without links.

use srvdb::types::DatabaseConfig;
use srvdb::{Error, Expansion, SrvDB, Vector, VectorEngine};
use std::path::Path;
use tempfile::TempDir;

const DIM: usize = 128;

fn vector(i: u64) -> Vector {
    Vector::new(
        (0..DIM)
            .map(|j| (((i * 7 + j as u64 * 3) % 11) as f32) - 5.0 + i as f32 * 0.01)
            .collect(),
    )
}

fn open(dir: &Path) -> SrvDB {
    SrvDB::new_with_config(dir.to_str().unwrap(), DatabaseConfig::new(DIM).unwrap()).unwrap()
}

/// A database with `n` records, persisted
fn with_records(n: u64) -> (TempDir, SrvDB) {
    let dir = TempDir::new().unwrap();
    let mut db = open(dir.path());
    let vectors: Vec<Vector> = (0..n).map(vector).collect();
    let metas: Vec<String> = (0..n).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    db.add_batch(&vectors, &metas).unwrap();
    db.persist().unwrap();
    (dir, db)
}

fn links(db: &SrvDB, id: u64) -> Vec<(String, u64)> {
    db.neighbors(id, None).unwrap()
}

#[test]
fn persisted_links_survive_reopen_and_unpersisted_ones_do_not() {
    let (dir, mut db) = with_records(5);
    assert!(db.link(0, "cites", 1).unwrap());
    assert!(!db.link(0, "cites", 1).unwrap());
    db.link(0, "about", 2).unwrap();
    db.persist().unwrap();
    db.link(3, "cites", 4).unwrap();
    db.unlink(0, "about", 2).unwrap();
    drop(db); // clean exit without persist

    let mut db = open(dir.path());
    assert_eq!(
        links(&db, 0),
        vec![("about".into(), 2), ("cites".into(), 1)]
    );
    assert!(links(&db, 3).is_empty());
    assert_eq!(db.link_count(), 2);
    assert_eq!(
        db.neighbors(0, Some("cites")).unwrap(),
        vec![("cites".into(), 1)]
    );

    db.unlink(0, "about", 2).unwrap();
    db.persist().unwrap();
    drop(db);
    assert_eq!(links(&open(dir.path()), 0), vec![("cites".into(), 1)]);
}

#[test]
fn links_to_records_added_in_the_same_persist_are_atomic() {
    let (dir, mut db) = with_records(2);
    let id = db.add(&vector(9), "{}").unwrap();
    db.link(0, "cites", id).unwrap();
    db.persist().unwrap();
    drop(db);
    assert_eq!(links(&open(dir.path()), 0), vec![("cites".into(), id)]);
}

#[test]
fn deleting_a_record_removes_its_links_durably() {
    let (dir, mut db) = with_records(4);
    db.link(0, "cites", 1).unwrap();
    db.link(1, "cites", 2).unwrap();
    db.link(3, "cites", 2).unwrap();
    db.delete(1).unwrap();
    assert!(links(&db, 0).is_empty(), "incoming link removed at once");
    assert!(db.neighbors(1, None).is_err());
    assert!(db.link(1, "cites", 3).is_err());
    db.persist().unwrap();
    drop(db);
    let db = open(dir.path());
    assert_eq!(db.link_count(), 1);
    assert_eq!(links(&db, 3), vec![("cites".into(), 2)]);
}

#[test]
fn unpersisted_delete_keeps_links_after_reopen() {
    let (dir, mut db) = with_records(3);
    db.link(0, "cites", 1).unwrap();
    db.persist().unwrap();
    db.delete(1).unwrap();
    drop(db);
    assert_eq!(links(&open(dir.path()), 0), vec![("cites".into(), 1)]);
}

#[test]
fn replace_moves_links_to_the_new_record() {
    let (dir, mut db) = with_records(3);
    db.link(0, "cites", 1).unwrap();
    db.link(1, "cites", 2).unwrap();
    let new_id = db.replace(1, &vector(1), "{}").unwrap();
    assert_eq!(links(&db, 0), vec![("cites".into(), new_id)]);
    assert_eq!(links(&db, new_id), vec![("cites".into(), 2)]);
    db.persist().unwrap();
    drop(db);
    let db = open(dir.path());
    assert_eq!(links(&db, 0), vec![("cites".into(), new_id)]);
    assert_eq!(db.link_count(), 2);
}

#[test]
fn compaction_renumbers_links_and_survives_reopen() {
    let (dir, mut db) = with_records(6);
    db.link(5, "cites", 3).unwrap();
    db.link(3, "next", 4).unwrap();
    db.link(0, "cites", 1).unwrap();
    db.delete(1).unwrap();
    db.delete(2).unwrap();
    assert_eq!(db.compact().unwrap(), 2);
    // old ids 0,3,4,5 -> 0,1,2,3
    let expect = |db: &SrvDB| {
        assert_eq!(links(db, 3), vec![("cites".into(), 1)]);
        assert_eq!(links(db, 1), vec![("next".into(), 2)]);
        assert!(links(db, 0).is_empty());
        assert_eq!(db.link_count(), 2);
    };
    expect(&db);
    drop(db);
    expect(&open(dir.path()));
}

#[test]
fn invalid_links_are_rejected_before_anything_changes() {
    let (_dir, mut db) = with_records(3);
    let invalid = |e: Error| assert!(matches!(e, Error::InvalidInput(_)), "{e}");
    invalid(db.link(0, "Cites", 1).unwrap_err());
    invalid(db.link(0, "", 1).unwrap_err());
    invalid(db.link(0, "cites", 3).unwrap_err());
    invalid(db.link(3, "cites", 0).unwrap_err());
    invalid(db.link(1, "cites", 1).unwrap_err());
    invalid(db.neighbors(7, None).unwrap_err());
    invalid(db.unlink(0, "bad type", 1).unwrap_err());
    assert_eq!(db.link_count(), 0);
}

#[test]
fn fan_out_limit_holds() {
    let (_dir, mut db) = with_records(1100);
    for t in 1..=1024 {
        db.link(0, "x", t).unwrap();
    }
    let err = db.link(0, "y", 1050).unwrap_err();
    assert!(err.to_string().contains("1024 outgoing links"), "{err}");
    db.unlink(0, "x", 1).unwrap();
    assert!(db.link(0, "y", 1050).unwrap());
}

#[test]
fn search_expand_marks_hops_and_respects_bounds() {
    let (_dir, mut db) = with_records(20);
    for i in 0..19 {
        db.link(i, "next", i + 1).unwrap();
    }
    let hits = db.search(&vector(5), 1).unwrap();
    let got = db
        .search_expand(&vector(5), 1, &Expansion::new("next", 3))
        .unwrap();
    let ids: Vec<(u64, u32)> = got.iter().map(|r| (r.id, r.hop)).collect();
    let first = hits[0].id;
    assert_eq!(
        ids,
        vec![(first, 0), (first + 1, 1), (first + 2, 2), (first + 3, 3)]
    );
    assert!(got
        .iter()
        .all(|r| r.score == hits[0].score && r.metadata.is_some()));

    let few = Expansion::new("next", 3).with_max_results(2);
    assert_eq!(db.search_expand(&vector(5), 1, &few).unwrap().len(), 3);
    for bad in [Expansion::new("next", 0), Expansion::new("next", 4)] {
        assert!(db.search_expand(&vector(5), 1, &bad).is_err());
    }
    // Other link types are not followed
    let none = db
        .search_expand(&vector(5), 1, &Expansion::new("cites", 2))
        .unwrap();
    assert_eq!(none.len(), 1);
}

/// A SrvDB 0.3 database has no links table: it opens with no links,
/// accepts links and keeps them across reopen.
#[test]
fn database_from_0_3_without_links_migrates() {
    let dir = TempDir::new().unwrap();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/v0.3/hnsw");
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dir.path().join(entry.file_name())).unwrap();
    }
    let path = dir.path().to_str().unwrap();
    {
        let mut db = SrvDB::open_existing(path).unwrap();
        assert_eq!(db.count(), 40);
        assert_eq!(db.link_count(), 0);
        db.link(0, "cites", 39).unwrap();
        db.persist().unwrap();
    }
    let db = SrvDB::open_existing(path).unwrap();
    assert_eq!(db.count(), 40);
    assert_eq!(links(&db, 0), vec![("cites".into(), 39)]);
}
