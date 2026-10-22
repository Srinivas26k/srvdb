//! Databases created with v0.2 keep working: their quantizer files (written
//! with bincode, before the versioned format) load through the public API.
//! Fixtures in tests/data/v0.2 were produced by v0.2's serializer.

use srvdb::{QuantizedVectorStorage, ScalarQuantizedStorage};
use tempfile::TempDir;

fn dir_with(file: &str) -> TempDir {
    let dir = TempDir::new().unwrap();
    std::fs::copy(
        format!("{}/tests/data/v0.2/{file}", env!("CARGO_MANIFEST_DIR")),
        dir.path().join(file),
    )
    .unwrap();
    dir
}

#[test]
fn v02_scalar_quantizer_loads_and_encodes() {
    let dir = dir_with("scalar_quantizer.bin");
    let mut storage = ScalarQuantizedStorage::load(dir.path().to_str().unwrap(), 4).unwrap();
    // Range per dimension from the fixture: [-1,1] [-0.5,0.5] [0,2] [0.25,0.75]
    let id = storage.append(&[1.0, -0.5, 1.0, 0.5]).unwrap();
    let decoded = storage.get(id).unwrap();
    let expected = [1.0, -0.5, 1.0, 0.5];
    for (got, want) in decoded.iter().zip(expected) {
        assert!((got - want).abs() < 0.01, "{decoded:?}");
    }
}

#[test]
fn v02_codebooks_load_and_encode() {
    let dir = dir_with("codebooks.bin");
    let mut storage = QuantizedVectorStorage::load(dir.path().to_str().unwrap()).unwrap();
    assert_eq!(storage.q_size, 2);
    // Nearest centroids in the fixture: [1,0] is #1, [0.25,0.75] is #2
    let id = storage.append(&[1.0, 0.0, 0.25, 0.75]).unwrap();
    assert_eq!(storage.get(id).unwrap(), &[1, 2]);
}

#[test]
fn v02_hnsw_graph_is_rebuilt_from_vectors() {
    use srvdb::{HNSWConfig, SrvDB, Vector, VectorEngine};
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let vectors: Vec<Vector> = (0..200)
        .map(|i| {
            Vector::new(
                (0..128)
                    .map(|j| ((i * 131 + j * 17) % 211) as f32 - 105.0)
                    .collect(),
            )
        })
        .collect();
    let metas: Vec<String> = (0..200).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    {
        let mut db = SrvDB::new_with_hnsw(path, 128, HNSWConfig::default()).unwrap();
        db.add_batch(&vectors, &metas).unwrap();
        db.persist().unwrap();
    }
    // A graph file without the versioned envelope, as v0.2 wrote it
    std::fs::write(dir.path().join("hnsw.graph"), b"bincode-encoded graph").unwrap();

    let db = SrvDB::new_with_hnsw(path, 128, HNSWConfig::default()).unwrap();
    assert_eq!(db.count(), 200);
    assert_eq!(db.search(&vectors[42], 1).unwrap()[0].id, 42);
}
