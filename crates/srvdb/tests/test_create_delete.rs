//! SrvDB::create, open_existing by path and delete_database (ADR 0013)

use srvdb::{CreateOptions, Error, IndexChoice, SrvDB, Vector, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;

type Case = (IndexChoice, &'static [(&'static str, u64)], &'static str);

fn vectors(n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..DIM)
                .map(|j| ((i * 7 + j * 3) % 11) as f32 + 1.0)
                .collect()
        })
        .collect()
}

fn add_some(db: &mut SrvDB, n: usize) {
    let vecs: Vec<Vector> = vectors(n).into_iter().map(Vector::new).collect();
    let metas = vec!["{}".to_string(); n];
    db.add_batch(&vecs, &metas).unwrap();
    db.persist().unwrap();
}

#[test]
fn every_index_reopens_with_only_the_path() {
    let cases: [Case; 6] = [
        (IndexChoice::Flat, &[], "flat"),
        (IndexChoice::Hnsw, &[("m", 8), ("ef_search", 20)], "hnsw"),
        (IndexChoice::Ivf, &[("nlist", 4), ("nprobe", 2)], "ivf"),
        (IndexChoice::Sq8, &[], "sq8"),
        (IndexChoice::Rabitq, &[("bits", 2), ("rerank", 5)], "rabitq"),
        (IndexChoice::Auto, &[], "auto"),
    ];
    for (index, params, name) in cases {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("db");
        let path = path.to_str().unwrap();
        let mut options = CreateOptions::new(index);
        for (key, value) in params {
            options.set(key, *value).unwrap();
        }
        let mut db = SrvDB::create(path, DIM, &options).unwrap();
        assert_eq!(db.index_name(), name);
        add_some(&mut db, 40);
        drop(db);
        let db = SrvDB::open_existing(path).unwrap();
        assert_eq!(db.index_name(), name, "{index}");
        assert_eq!(db.count(), 40, "{index}");
        if index == IndexChoice::Rabitq {
            let config = db.rabitq_config().unwrap();
            assert_eq!((config.bits, config.rerank), (2, 5));
        }
    }
}

#[test]
fn quantizers_train_on_given_vectors() {
    for index in [IndexChoice::Sq8, IndexChoice::Pq] {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();
        let mut options = CreateOptions::new(index);
        options.set_training(vectors(300)).unwrap();
        let mut db = SrvDB::create(path, DIM, &options).unwrap();
        add_some(&mut db, 10);
        drop(db);
        assert_eq!(SrvDB::open_existing(path).unwrap().count(), 10);
    }
}

#[test]
fn create_refuses_existing_paths_and_bad_options() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    // An empty directory is fine
    drop(SrvDB::create(path, DIM, &CreateOptions::default()).unwrap());
    let err = SrvDB::create(path, DIM, &CreateOptions::default())
        .err()
        .unwrap();
    assert!(matches!(err, Error::AlreadyExists(_)), "{err:?}");
    assert!(err.to_string().contains("already exists"), "{err}");

    let other = TempDir::new().unwrap();
    std::fs::write(other.path().join("notes.txt"), "x").unwrap();
    let other_path = other.path().to_str().unwrap();
    let err = SrvDB::create(other_path, DIM, &CreateOptions::default())
        .err()
        .unwrap();
    assert!(err.to_string().contains("not empty"), "{err}");

    let fresh = dir.path().join("fresh");
    let fresh = fresh.to_str().unwrap();
    let mut options = CreateOptions::new(IndexChoice::Hnsw);
    options.set("m", 1).unwrap();
    assert!(matches!(
        SrvDB::create(fresh, DIM, &options),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        SrvDB::create(fresh, 0, &CreateOptions::default()),
        Err(Error::InvalidInput(_))
    ));
    let mut options = CreateOptions::new(IndexChoice::Sq8);
    options.set_training(vec![vec![1.0; DIM + 1]]).unwrap();
    assert!(matches!(
        SrvDB::create(fresh, DIM, &options),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(
        !std::path::Path::new(fresh).exists(),
        "bad options write nothing"
    );
}

#[test]
fn delete_removes_only_closed_srvdb_directories() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("db");
    let path = db_path.to_str().unwrap();
    let mut db = SrvDB::create(path, DIM, &CreateOptions::default()).unwrap();
    add_some(&mut db, 5);
    let err = SrvDB::delete_database(path).err().unwrap();
    assert!(matches!(err, Error::Locked(_)), "{err:?}");
    drop(db);

    std::fs::write(db_path.join("keep.txt"), "mine").unwrap();
    let err = SrvDB::delete_database(path).err().unwrap();
    assert!(err.to_string().contains("keep.txt"), "{err}");
    assert!(db_path.join("vectors.bin").exists());
    std::fs::remove_file(db_path.join("keep.txt")).unwrap();

    SrvDB::delete_database(path).unwrap();
    assert!(!db_path.exists());
    assert!(matches!(
        SrvDB::delete_database(path),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn delete_refuses_directories_without_srvdb_files() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    // Owned file names alone are not enough: the magic number is checked
    std::fs::write(dir.path().join("vectors.bin"), b"not a database").unwrap();
    let err = SrvDB::delete_database(path).err().unwrap();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    assert!(dir.path().join("vectors.bin").exists());
}

#[test]
fn exist_ok_opens_only_matching_databases() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let mut options = CreateOptions::new(IndexChoice::Hnsw);
    options.set("m", 8).unwrap().set_exist_ok(true);
    let mut db = SrvDB::create(path, DIM, &options).unwrap();
    add_some(&mut db, 3);
    drop(db);
    assert_eq!(SrvDB::create(path, DIM, &options).unwrap().count(), 3);

    options.set("m", 12).unwrap();
    let err = SrvDB::create(path, DIM, &options).err().unwrap();
    assert!(matches!(err, Error::SettingsMismatch(_)), "{err:?}");
    assert!(err.to_string().contains("m = 8"), "{err}");
    let mut flat = CreateOptions::new(IndexChoice::Flat);
    flat.set_exist_ok(true);
    let err = SrvDB::create(path, DIM, &flat).err().unwrap();
    assert!(err.to_string().contains("'hnsw', not 'flat'"), "{err}");
    let mut same = CreateOptions::new(IndexChoice::Hnsw);
    same.set_exist_ok(true);
    assert!(matches!(
        SrvDB::create(path, DIM + 1, &same),
        Err(Error::DimensionMismatch { .. })
    ));
}
