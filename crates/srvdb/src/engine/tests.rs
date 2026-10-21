//! Engine-level unit tests

use super::*;
use tempfile::TempDir;

#[test]
fn test_batch_operations() {
    let temp_dir = TempDir::new().unwrap();
    let mut db = SrvDB::new(temp_dir.path().to_str().unwrap()).unwrap();

    let vectors: Vec<Vector> = (0..100)
        .map(|i| Vector::new(vec![i as f32 / 100.0; 1536]))
        .collect();

    let metas: Vec<String> = (0..100).map(|i| format!(r#"{{"id": {}}}"#, i)).collect();

    let ids = db.add_batch(&vectors, &metas).unwrap();
    assert_eq!(ids.len(), 100);

    db.persist().unwrap();

    let results = db.search(&vectors[0], 5).unwrap();
    assert_eq!(results.len(), 5);
    assert!(results[0].score > 0.99);
}

#[test]
fn test_concurrent_search() {
    use std::sync::Arc;
    use std::thread;

    let temp_dir = TempDir::new().unwrap();
    let mut db = SrvDB::new(temp_dir.path().to_str().unwrap()).unwrap();

    // Add vectors
    let vectors: Vec<Vector> = (0..1000)
        .map(|_| Vector::new(vec![rand::random::<f32>(); 1536]))
        .collect();
    let metas: Vec<String> = (0..1000).map(|i| format!(r#"{{"id": {}}}"#, i)).collect();
    db.add_batch(&vectors, &metas).unwrap();
    db.persist().unwrap();

    let db = Arc::new(db);
    let query = Vector::new(vec![0.5; 1536]);

    // Spawn multiple search threads
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let db = Arc::clone(&db);
            let q = query.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    let _ = db.search(&q, 10);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn rejects_invalid_input_without_writing() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();
    let mut db = SrvDB::new_with_config(path, types::DatabaseConfig::new(128).unwrap()).unwrap();
    let good = Vector::new(vec![0.1; 128]);
    let mut nan = vec![0.1; 128];
    nan[5] = f32::NAN;

    let err = db.add(&Vector::new(vec![0.1; 64]), "{}").unwrap_err();
    assert!(matches!(err, Error::DimensionMismatch { .. }));
    let err = db.add(&Vector::new(nan.clone()), "{}").unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
    let err = db.add(&good, "not json").unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));

    // One bad item rejects the whole batch before anything is stored
    let batch = vec![good.clone(), Vector::new(nan)];
    let metas = vec!["{}".to_string(), "{}".to_string()];
    assert!(db.add_batch(&batch, &metas).is_err());
    assert!(db.add_batch(&batch[..1], &metas).is_err());
    assert_eq!(db.count(), 0);

    db.add(&good, "{}").unwrap();
    // Writes become visible to search on sync() (see the durability work)
    db.sync().unwrap();
    assert!(matches!(db.search(&good, 0), Err(Error::InvalidInput(_))));
    assert!(matches!(
        db.search(&Vector::new(vec![0.1; 3]), 1),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(db.search_batch(std::slice::from_ref(&good), 0).is_err());
    assert_eq!(db.search(&good, 1).unwrap().len(), 1);
}
