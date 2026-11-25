//! Quantized constructors enforce the dimension bounds of
//! `DatabaseConfig::new` (issue #48)

use srvdb::{Error, SrvDB, Vector};
use tempfile::TempDir;

fn training(dim: usize) -> Vec<Vec<f32>> {
    (0..8)
        .map(|i| (0..dim).map(|j| ((i * 31 + j) % 7) as f32 - 3.0).collect())
        .collect()
}

#[test]
fn scalar_quantized_rejects_small_dimension() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");
    let err = SrvDB::new_scalar_quantized(path.to_str().unwrap(), 64, &training(64))
        .err()
        .unwrap();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    assert!(
        !path.exists(),
        "nothing is created for a rejected dimension"
    );
}

#[test]
fn product_quantized_rejects_small_dimension() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");
    let vectors: Vec<Vector> = training(64).into_iter().map(Vector::new).collect();
    let err = SrvDB::new_quantized(path.to_str().unwrap(), &vectors)
        .err()
        .unwrap();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    assert!(
        !path.exists(),
        "nothing is created for a rejected dimension"
    );
}

#[test]
fn quantized_constructors_reject_oversized_dimension() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("db");
    let err = SrvDB::new_scalar_quantized(path.to_str().unwrap(), 4100, &training(4100))
        .err()
        .unwrap();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}
