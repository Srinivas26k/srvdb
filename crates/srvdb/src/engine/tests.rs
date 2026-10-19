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
