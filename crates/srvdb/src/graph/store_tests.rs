//! Validation of stored links on open (ADR 0016)

use super::*;
use crate::storage::metadata::MetadataStore;
use tempfile::TempDir;

fn store() -> (TempDir, MetadataStore) {
    let dir = TempDir::new().unwrap();
    let store = MetadataStore::new(dir.path().to_str().unwrap()).unwrap();
    (dir, store)
}

/// Write raw links and a format value, bypassing validation
fn write_raw(store: &MetadataStore, links: &[(u64, &str, u64)], format: u64) {
    let txn = store.database().begin_write().unwrap();
    {
        let mut table = txn.open_table(LINKS).unwrap();
        for &(s, t, d) in links {
            table.insert((s, t, d), ()).unwrap();
        }
        txn.open_table(STATE)
            .unwrap()
            .insert(FORMAT_KEY, format)
            .unwrap();
    }
    txn.commit().unwrap();
}

fn corrupt(store: &MetadataStore, count: u64) -> String {
    match load(store.database(), count, &HashSet::new()) {
        Err(e) => match crate::Error::from(e) {
            crate::Error::Corrupt(msg) => msg,
            other => panic!("expected Corrupt, got {other}"),
        },
        Ok(_) => panic!("expected Corrupt"),
    }
}

#[test]
fn missing_table_loads_empty() {
    let (_dir, store) = store();
    let graph = load(store.database(), 10, &HashSet::new()).unwrap();
    assert_eq!(graph.len(), 0);
}

#[test]
fn newer_format_is_rejected() {
    let (_dir, store) = store();
    write_raw(&store, &[], 2);
    assert!(corrupt(&store, 10).contains("format 2"));
}

#[test]
fn out_of_range_self_and_badly_named_links_are_rejected() {
    for bad in [(0, "a", 10), (10, "a", 0), (3, "a", 3), (0, "Bad", 1)] {
        let (_dir, store) = store();
        write_raw(&store, &[bad], FORMAT);
        assert!(corrupt(&store, 10).contains("invalid link"), "{bad:?}");
    }
}

#[test]
fn links_to_tombstoned_records_are_dropped_and_queued() {
    let (_dir, store) = store();
    write_raw(&store, &[(0, "a", 1), (2, "a", 1), (2, "a", 3)], FORMAT);
    let deleted: HashSet<u64> = [1].into();
    let graph = load(store.database(), 10, &deleted).unwrap();
    assert_eq!(graph.all(), vec![(2, "a".to_string(), 3)]);
    assert_eq!(graph.pending().len(), 2);
}
