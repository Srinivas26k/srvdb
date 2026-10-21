//! Product-quantized storage tests

use super::*;
use tempfile::TempDir;

#[test]
fn test_quantized_storage() {
    let temp_dir = TempDir::new().unwrap();

    // Create training data
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            let mut v = vec![0.0f32; 1536];
            v[0] = i as f32 / 100.0;
            v
        })
        .collect();

    // Convert training data to reference slice
    // let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let mut storage =
        QuantizedVectorStorage::new_with_training(temp_dir.path().to_str().unwrap(), &training)
            .unwrap();

    // Test append
    let test_vec = vec![0.5f32; 1536];
    let id = storage.append(&test_vec).unwrap();
    assert_eq!(id, 0);

    storage.flush().unwrap();

    // Test retrieval
    let qvec = storage.get(0).unwrap();
    assert_eq!(qvec.len(), 192);

    // Test stats
    let stats = storage.get_stats();
    assert_eq!(stats.vector_count, 1);
    assert!(stats.compression_ratio > 30.0); // ~32x compression
}
