//! Graph links (ADR 0016) against changes made by a SrvDB without links,
//! simulated through `MetadataStore` (which is what SrvDB 0.3 commits and
//! compacts with), plus replace leaving no orphan on error.

use srvdb::types::DatabaseConfig;
use srvdb::{Error, MetadataStore, SrvDB, Vector, VectorEngine};
use std::path::Path;
use tempfile::TempDir;

const DIM: usize = 128;

fn vector(i: u64) -> Vector {
    Vector::new(
        (0..DIM)
            .map(|j| ((i * 13 + j as u64) % 17) as f32)
            .collect(),
    )
}

fn open(dir: &Path) -> srvdb::Result<SrvDB> {
    SrvDB::new_with_config(dir.to_str().unwrap(), DatabaseConfig::new(DIM).unwrap())
}

/// Six records, links 5->3 and 0->1, record 2 deleted; all persisted
fn linked_db() -> TempDir {
    let dir = TempDir::new().unwrap();
    let mut db = open(dir.path()).unwrap();
    let vectors: Vec<Vector> = (0..6).map(vector).collect();
    db.add_batch(&vectors, &vec!["{}".to_string(); 6]).unwrap();
    db.link(5, "cites", 3).unwrap();
    db.link(0, "cites", 1).unwrap();
    db.delete(2).unwrap();
    db.persist().unwrap();
    dir
}

fn expect_stale(dir: &Path) {
    match open(dir) {
        Err(Error::Corrupt(msg)) => assert!(msg.contains("drop_links"), "{msg}"),
        Err(e) => panic!("expected Corrupt, got {e}"),
        Ok(_) => panic!("stale links were accepted"),
    }
}

/// SrvDB 0.3 compacts with MetadataStore::compact, which renumbers
/// metadata but leaves the links table alone. Opening must refuse the
/// now-wrong links instead of using or silently dropping them.
#[test]
fn compaction_by_an_older_version_is_detected_and_recoverable() {
    let dir = linked_db();
    {
        let mut store = MetadataStore::new(dir.path().to_str().unwrap()).unwrap();
        store.compact(&[0, 1, 3, 4, 5]).unwrap();
        store.finish_compaction().unwrap();
    }
    expect_stale(dir.path());

    let path = dir.path().to_str().unwrap();
    assert_eq!(SrvDB::drop_links(path).unwrap(), 2);
    let db = open(dir.path()).unwrap();
    assert_eq!(db.link_count(), 0);
    assert_eq!(db.live_count(), 5, "records are untouched");
}

/// A delete committed by an older version changes the tombstones
#[test]
fn delete_by_an_older_version_is_detected() {
    let dir = linked_db();
    {
        let mut store = MetadataStore::new(dir.path().to_str().unwrap()).unwrap();
        let count = store.committed_count().unwrap().unwrap();
        store.commit(count, &[3]).unwrap();
    }
    expect_stale(dir.path());
}

/// A database whose own commits carry the stamp reopens normally, also
/// after commits without link changes and after compaction
#[test]
fn own_commits_keep_the_stamp_current() {
    let dir = linked_db();
    {
        let mut db = open(dir.path()).unwrap();
        db.add(&vector(9), "{}").unwrap();
        db.persist().unwrap(); // no link changes
        db.compact().unwrap();
    }
    let db = open(dir.path()).unwrap();
    assert_eq!(db.link_count(), 2);
}

#[test]
fn drop_links_needs_an_existing_closed_database() {
    let dir = linked_db();
    let missing = dir.path().join("missing");
    assert!(matches!(
        SrvDB::drop_links(missing.to_str().unwrap()),
        Err(Error::NotFound(_))
    ));
    let _db = open(dir.path()).unwrap();
    assert!(matches!(
        SrvDB::drop_links(dir.path().to_str().unwrap()),
        Err(Error::Locked(_))
    ));
}

/// replace validates (and checks link limits) before writing: a failed
/// replace adds no record and keeps the links; a record at the outgoing
/// limit can still be replaced
#[test]
fn replace_never_leaves_an_orphan() {
    let dir = TempDir::new().unwrap();
    let mut db = open(dir.path()).unwrap();
    let vectors: Vec<Vector> = (0..1100).map(vector).collect();
    db.add_batch(&vectors, &vec!["{}".to_string(); 1100])
        .unwrap();
    for t in 1..=1024 {
        db.link(0, "x", t).unwrap();
    }
    db.link(1050, "x", 0).unwrap();
    let before = db.count();
    let bad = Vector::new(vec![f32::NAN; DIM]);
    assert!(db.replace(0, &bad, "{}").is_err());
    assert_eq!(db.count(), before, "no orphaned record");
    assert_eq!(db.neighbors(0, None).unwrap().len(), 1024);

    let new_id = db.replace(0, &vector(0), "{}").unwrap();
    assert_eq!(db.count(), before + 1);
    assert_eq!(db.neighbors(new_id, None).unwrap().len(), 1024);
    assert_eq!(
        db.neighbors(1050, None).unwrap(),
        vec![("x".into(), new_id)]
    );
    assert_eq!(db.link_count(), 1025);
}
