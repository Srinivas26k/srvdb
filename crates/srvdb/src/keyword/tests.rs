//! Engine-level keyword tests that need crate internals (audit fixes)

use super::search::{MAX_TERMS, MAX_TEXT_BYTES};
use crate::error::Error;
use crate::storage::settings::{self, IndexKind};
use crate::{DatabaseConfig, HNSWConfig, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;

fn vector(i: usize) -> Vector {
    Vector::new((0..DIM).map(|j| ((i + j) % 11) as f32 + 1.0).collect())
}

fn flat(dir: &TempDir) -> SrvDB {
    let path = dir.path().to_str().unwrap();
    SrvDB::new_with_config(path, DatabaseConfig::new(DIM).unwrap()).unwrap()
}

#[test]
fn query_text_is_bounded_before_the_index_is_touched() {
    let dir = TempDir::new().unwrap();
    // Keyword search is not enabled: a bound violation must still be
    // InvalidInput, proving the check runs before the index is looked at
    let db = flat(&dir);
    let long = "a ".repeat(MAX_TEXT_BYTES / 2 + 1);
    assert!(matches!(
        db.keyword_search(&long, 1, None),
        Err(Error::InvalidInput(_))
    ));
    let many: String = (0..=MAX_TERMS).map(|i| format!("w{i} ")).collect();
    assert!(many.len() <= MAX_TEXT_BYTES);
    assert!(matches!(
        db.keyword_search(&many, 1, None),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        db.search_hybrid(&vector(0), &many, 1, None),
        Err(Error::InvalidInput(_))
    ));
    let at_limit: String = (0..MAX_TERMS).map(|i| format!("w{i} ")).collect();
    assert!(matches!(
        db.keyword_search(&at_limit, 1, None),
        Err(Error::InvalidState(_))
    ));
}

#[test]
fn deletes_after_loading_are_applied_without_rescanning() {
    let dir = TempDir::new().unwrap();
    let mut db = flat(&dir);
    let metas: Vec<String> = (0..3)
        .map(|i| format!(r#"{{"text": "fox {i}"}}"#))
        .collect();
    db.add_batch(&(0..3).map(vector).collect::<Vec<_>>(), &metas)
        .unwrap();
    db.enable_keyword_index("text").unwrap();
    db.delete(0).unwrap();
    db.persist().unwrap();
    drop(db);

    let mut db = flat(&dir);
    assert_eq!(db.keyword_search("fox", 5, None).unwrap().len(), 2); // loads
    db.delete(1).unwrap();
    let ids: Vec<u64> = db
        .keyword_search("fox", 5, None)
        .unwrap()
        .iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(ids, [2]);
    let super::Slot::On { index, .. } = &*db.keyword.0.lock() else {
        panic!("keyword index not loaded");
    };
    assert_eq!(index.docs(), 1);
}

#[cfg(unix)]
#[test]
fn enable_returns_io_errors_instead_of_rebuilding() {
    let dir = TempDir::new().unwrap();
    let mut db = flat(&dir);
    std::fs::create_dir(dir.path().join(super::FILE_NAME)).unwrap();
    let err = db.enable_keyword_index("text").unwrap_err();
    assert!(matches!(err, Error::Io(_)), "{err:?}");
}

#[cfg(unix)]
#[test]
fn settings_are_recorded_even_if_the_keyword_write_fails() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let mut db = SrvDB::new_with_hnsw(path, DIM, HNSWConfig::new(8, 40, 20)).unwrap();
    db.add(&vector(0), r#"{"text": "fox"}"#).unwrap();
    db.enable_keyword_index("text").unwrap();
    db.set_ef_search(77).unwrap();
    // write_atomic cannot remove a directory at its temp path
    let tmp = dir.path().join(format!(".{}.tmp", super::FILE_NAME));
    std::fs::create_dir(&tmp).unwrap();
    assert!(db.persist().is_err());
    let stored = settings::load(dir.path()).unwrap().unwrap();
    let IndexKind::Hnsw(p) = stored.kind else {
        panic!("expected hnsw settings");
    };
    assert_eq!(p.ef_search, 77);
}
