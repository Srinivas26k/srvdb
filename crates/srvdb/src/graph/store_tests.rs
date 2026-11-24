//! Validation of stored links on open (ADR 0016)

use super::*;
use crate::storage::metadata::MetadataStore;
use tempfile::TempDir;

fn store() -> (TempDir, MetadataStore) {
    let dir = TempDir::new().unwrap();
    let store = MetadataStore::new(dir.path().to_str().unwrap()).unwrap();
    (dir, store)
}

/// Write raw links, a format and the stamp of (`count`, no tombstones),
/// bypassing validation
fn write_raw(store: &MetadataStore, links: &[(u64, &str, u64)], format: u64, count: u64) {
    let txn = store.database().begin_write().unwrap();
    {
        let mut table = txn.open_table(LINKS).unwrap();
        for &(s, t, d) in links {
            table.insert((s, t, d), ()).unwrap();
        }
        let mut state = txn.open_table(STATE).unwrap();
        state.insert(FORMAT_KEY, format).unwrap();
        state.insert(STAMP_KEY, stamp(count, [])).unwrap();
    }
    txn.commit().unwrap();
}

fn load_with(store: &MetadataStore, count: u64, deleted: &HashSet<u64>, limits: Limits) -> String {
    match load(store.database(), count, deleted, limits) {
        Err(e) => match crate::Error::from(e) {
            crate::Error::Corrupt(msg) => msg,
            other => panic!("expected Corrupt, got {other}"),
        },
        Ok(_) => panic!("expected Corrupt"),
    }
}

fn corrupt(store: &MetadataStore, count: u64) -> String {
    load_with(store, count, &HashSet::new(), Limits::DEFAULT)
}

#[test]
fn missing_table_loads_empty() {
    let (_dir, store) = store();
    let graph = load(store.database(), 10, &HashSet::new(), Limits::DEFAULT).unwrap();
    assert_eq!(graph.len(), 0);
}

#[test]
fn newer_format_is_rejected() {
    let (_dir, store) = store();
    write_raw(&store, &[], 2, 10);
    assert!(corrupt(&store, 10).contains("format 2"));
}

#[test]
fn out_of_range_self_and_badly_named_links_are_rejected() {
    for bad in [(0, "a", 10), (10, "a", 0), (3, "a", 3), (0, "Bad", 1)] {
        let (_dir, store) = store();
        write_raw(&store, &[bad], FORMAT, 10);
        assert!(corrupt(&store, 10).contains("invalid link"), "{bad:?}");
    }
}

#[test]
fn links_touching_tombstones_are_refused_not_dropped() {
    let (_dir, store) = store();
    let deleted: HashSet<u64> = [1].into();
    let txn = store.database().begin_write().unwrap();
    {
        txn.open_table(LINKS)
            .unwrap()
            .insert((0, "a", 1), ())
            .unwrap();
        let mut state = txn.open_table(STATE).unwrap();
        state.insert(FORMAT_KEY, FORMAT).unwrap();
        state.insert(STAMP_KEY, stamp(10, [1])).unwrap();
    }
    txn.commit().unwrap();
    let msg = load_with(&store, 10, &deleted, Limits::DEFAULT);
    assert!(msg.contains("invalid link"), "{msg}");
}

#[test]
fn stale_or_missing_stamp_is_refused_with_the_recovery_call() {
    let (_dir, store) = store();
    write_raw(&store, &[(0, "a", 1)], FORMAT, 10);
    assert!(corrupt(&store, 11).contains("drop_links"), "count changed");
    let tomb: HashSet<u64> = [5].into();
    let msg = load_with(&store, 10, &tomb, Limits::DEFAULT);
    assert!(msg.contains("drop_links"), "tombstones changed: {msg}");
    assert_eq!(
        load(store.database(), 10, &HashSet::new(), Limits::DEFAULT)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn limits_are_enforced_while_loading() {
    let tight = Limits {
        out_per_record: 2,
        in_per_record: 2,
        total: 3,
        types: 1,
    };
    let cases: [&[(u64, &str, u64)]; 4] = [
        &[(0, "a", 1), (0, "a", 2), (0, "a", 3)],
        &[(1, "a", 0), (2, "a", 0), (3, "a", 0)],
        &[(0, "a", 1), (1, "a", 2), (2, "a", 3), (3, "a", 4)],
        &[(0, "a", 1), (0, "b", 2)],
    ];
    for links in cases {
        let (_dir, store) = store();
        write_raw(&store, links, FORMAT, 10);
        let msg = load_with(&store, 10, &HashSet::new(), tight);
        assert!(msg.contains("more links than allowed"), "{links:?}: {msg}");
    }
}

#[test]
fn stamp_depends_on_count_and_tombstone_set() {
    assert_eq!(stamp(10, [1, 2]), stamp(10, [2, 1]));
    assert_ne!(stamp(10, [1, 2]), stamp(10, [1, 3]));
    assert_ne!(stamp(10, [1]), stamp(10, []));
    assert_ne!(stamp(10, []), stamp(11, []));
}

#[test]
fn clear_removes_links_and_restamps() {
    let (_dir, store) = store();
    write_raw(&store, &[(0, "a", 1), (1, "a", 2)], FORMAT, 99);
    assert_eq!(clear(store.database()).unwrap(), 2);
    let graph = load(store.database(), 0, &HashSet::new(), Limits::DEFAULT).unwrap();
    assert_eq!(graph.len(), 0);
}
