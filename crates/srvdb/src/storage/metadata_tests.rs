//! MetadataStore tests

use super::*;
use tempfile::TempDir;

fn store(dir: &TempDir) -> MetadataStore {
    MetadataStore::new(dir.path().to_str().unwrap()).unwrap()
}

#[test]
fn set_get_delete_and_list() {
    let dir = TempDir::new().unwrap();
    let mut store = store(&dir);
    store.set(1, r#"{"a": 1}"#).unwrap();
    store
        .set_batch(&[(2, "{}".to_string()), (3, "[]".to_string())])
        .unwrap();
    assert_eq!(store.get(1).unwrap().as_deref(), Some(r#"{"a": 1}"#));
    assert_eq!(
        store.get_batch(&[3, 9]).unwrap(),
        vec![Some("[]".to_string()), None]
    );

    store.delete(2).unwrap();
    assert_eq!(store.count().unwrap(), 2);
    let ids: Vec<u64> = store.all().unwrap().into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, vec![1, 3]);
}

#[test]
fn committed_writes_survive_reopen() {
    let dir = TempDir::new().unwrap();
    {
        let mut store = store(&dir);
        store.set(0, "{}").unwrap();
        store.commit(1, &[]).unwrap();
    }
    let store = store(&dir);
    assert_eq!(store.committed_count().unwrap(), Some(1));
    assert_eq!(store.get(0).unwrap().as_deref(), Some("{}"));
}

#[test]
fn database_without_commit_record_reports_none() {
    let dir = TempDir::new().unwrap();
    assert_eq!(store(&dir).committed_count().unwrap(), None);
}

#[test]
fn truncate_from_removes_uncommitted_ids() {
    let dir = TempDir::new().unwrap();
    let mut store = store(&dir);
    for id in 0..5 {
        store.set(id, "{}").unwrap();
    }
    store.truncate_from(3).unwrap();
    let ids: Vec<u64> = store.all().unwrap().into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, vec![0, 1, 2]);
}

/// Write metadata.db the way SrvDB 0.2 did: redb 2.x file format v2, one
/// "metadata" table, no commit record
pub(crate) fn write_v02_metadata(dir: &std::path::Path, entries: &[(u64, &str)]) {
    const V02_TABLE: redb2::TableDefinition<u64, &str> = redb2::TableDefinition::new("metadata");
    let db = redb2::Database::create(dir.join("metadata.db")).unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(V02_TABLE).unwrap();
        for (id, meta) in entries {
            table.insert(*id, *meta).unwrap();
        }
    }
    txn.commit().unwrap();
}

#[test]
fn metadata_written_by_redb_2_is_upgraded_on_open() {
    let dir = TempDir::new().unwrap();
    write_v02_metadata(dir.path(), &[(0, r#"{"a": 0}"#), (7, r#"{"a": 7}"#)]);

    let store = store(&dir);
    assert_eq!(store.get(7).unwrap().as_deref(), Some(r#"{"a": 7}"#));
    assert_eq!(store.count().unwrap(), 2);
    assert_eq!(store.committed_count().unwrap(), None);
    drop(store);
    // Already upgraded: opens directly the second time
    assert_eq!(
        MetadataStore::new(dir.path().to_str().unwrap())
            .unwrap()
            .count()
            .unwrap(),
        2
    );
}
