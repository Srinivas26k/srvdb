//! Failure and race safety of SrvDB::create and delete_database (ADR 0013)

use srvdb::{CreateOptions, Error, IndexChoice, SrvDB, VectorEngine};
use tempfile::TempDir;

const DIM: usize = 128;

fn training(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..dim).map(|j| ((i * 5 + j) % 9) as f32 + 1.0).collect())
        .collect()
}

/// Names in `dir` other than `keep`
fn leftovers(dir: &std::path::Path, keep: &str) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n != keep)
        .collect()
}

#[test]
fn failed_create_leaves_nothing_and_a_retry_succeeds() {
    let parent = TempDir::new().unwrap();
    let target = parent.path().join("db");
    let path = target.to_str().unwrap();
    // PQ needs a dimension divisible by 4; the error comes from training,
    // after the lock and metadata files were made
    let mut bad = CreateOptions::new(IndexChoice::Pq);
    bad.set_training(training(300, 130)).unwrap();
    let err = SrvDB::create(path, 130, &bad).err().unwrap();
    assert!(err.to_string().contains("divisible by 4"), "{err}");
    assert!(!target.exists(), "nothing at the path after a failure");
    assert!(leftovers(parent.path(), "db").is_empty(), "no staging left");

    let mut good = CreateOptions::new(IndexChoice::Pq);
    good.set_training(training(300, DIM)).unwrap();
    let db = SrvDB::create(path, DIM, &good).unwrap();
    assert_eq!(db.count(), 0);
    assert_eq!(leftovers(parent.path(), "db"), Vec::<String>::new());
}

#[test]
fn create_into_an_existing_empty_directory() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let db = SrvDB::create(path, DIM, &CreateOptions::default()).unwrap();
    drop(db);
    assert!(SrvDB::open_existing(path).is_ok());
}

#[test]
fn delete_leaves_no_tombstone_and_frees_the_path() {
    let parent = TempDir::new().unwrap();
    let target = parent.path().join("db");
    let path = target.to_str().unwrap();
    drop(SrvDB::create(path, DIM, &CreateOptions::default()).unwrap());
    SrvDB::delete_database(path).unwrap();
    assert!(leftovers(parent.path(), "").is_empty(), "no tombstone left");
    // The path is free for a new database
    drop(SrvDB::create(path, DIM, &CreateOptions::default()).unwrap());
}

#[cfg(unix)]
#[test]
fn delete_refuses_a_symlink_to_a_database() {
    let parent = TempDir::new().unwrap();
    let target = parent.path().join("db");
    drop(SrvDB::create(target.to_str().unwrap(), DIM, &CreateOptions::default()).unwrap());
    let link = parent.path().join("link");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let err = SrvDB::delete_database(link.to_str().unwrap())
        .err()
        .unwrap();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    assert!(err.to_string().contains("symbolic link"), "{err}");
    assert!(target.join("vectors.bin").exists());
}

#[cfg(unix)]
#[test]
fn a_fifo_named_like_a_srvdb_file_does_not_hang() {
    let dir = TempDir::new().unwrap();
    let fifo = dir.path().join("index_settings.bin");
    let status = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap();
    assert!(status.success());
    let path = dir.path().to_str().unwrap();
    // Opening the FIFO for reading would block forever
    let err = SrvDB::create(path, DIM, &CreateOptions::default())
        .err()
        .unwrap();
    assert!(err.to_string().contains("not empty"), "{err}");
    let err = SrvDB::delete_database(path).err().unwrap();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}
