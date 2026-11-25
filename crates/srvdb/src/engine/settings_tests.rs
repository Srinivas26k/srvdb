//! Every mode reopens through `open_existing` with its settings (ADR 0011)

use super::*;
use crate::storage::settings::{HnswParams, IndexKind, FILE_NAME};
use crate::RabitqConfig;
use tempfile::TempDir;

const DIM: usize = 128;
const N: usize = 300;

fn vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|j| ((i * 7 + j * 13) % 29) as f32 - 14.0 + (i as f32) * 0.01)
        .collect()
}

fn vectors() -> Vec<Vector> {
    (0..N).map(|i| Vector::new(vector(i))).collect()
}

fn fill(db: &mut SrvDB) {
    let metas: Vec<String> = (0..N).map(|i| format!(r#"{{"i": {i}}}"#)).collect();
    db.add_batch(&vectors(), &metas).unwrap();
    db.persist().unwrap();
}

fn hnsw_config() -> hnsw::HNSWConfig {
    hnsw::HNSWConfig::new(12, 120, 70)
}

/// Create a database in `mode` with non-default parameters
fn create(path: &str, mode: &str) -> SrvDB {
    let training: Vec<Vec<f32>> = (0..N).map(vector).collect();
    let mut db = match mode {
        "flat" => SrvDB::new_with_config(path, types::DatabaseConfig::new(DIM).unwrap()).unwrap(),
        "hnsw" => SrvDB::new_with_hnsw(path, DIM, hnsw_config()).unwrap(),
        "sq8" => SrvDB::new_scalar_quantized(path, DIM, &training).unwrap(),
        "pq" => SrvDB::new_quantized(path, &vectors()).unwrap(),
        "hnsw_pq" => SrvDB::new_with_hnsw_quantized(path, &vectors(), hnsw_config()).unwrap(),
        "rabitq" => {
            let config = RabitqConfig::new(2).unwrap().with_rerank(7).with_seed(9);
            SrvDB::new_rabitq(path, DIM, config).unwrap()
        }
        "ivf" | "auto" => {
            SrvDB::new_with_config(path, types::DatabaseConfig::new(DIM).unwrap()).unwrap()
        }
        other => panic!("unknown mode {other}"),
    };
    if mode == "ivf" {
        let config = ivf::IVFConfig {
            nlist: 8,
            nprobe: 3,
            ..Default::default()
        };
        db.configure_ivf(config).unwrap();
        db.set_mode(IndexMode::Ivf);
    }
    if mode == "auto" {
        db.set_mode(IndexMode::Auto);
    }
    fill(&mut db);
    if mode == "ivf" {
        db.train_ivf().unwrap();
    }
    if mode == "hnsw" {
        db.set_ef_search(90).unwrap(); // search-time knob, recorded by persist
    }
    db.persist().unwrap();
    db
}

const MODES: [&str; 8] = [
    "flat", "hnsw", "sq8", "pq", "hnsw_pq", "rabitq", "ivf", "auto",
];

#[test]
fn every_mode_reopens_with_its_mode_and_parameters() {
    for mode in MODES {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();
        let expected = create(path, mode).current_settings();
        let db = SrvDB::open_existing(path).unwrap();
        assert_eq!(db.current_settings(), expected, "{mode}");
        assert_eq!(db.count(), N as u64, "{mode}");
        let hit = db.search(&Vector::new(vector(5)), 1).unwrap();
        assert_eq!(hit.len(), 1, "{mode}");
        match (mode, expected.kind) {
            ("flat", IndexKind::Flat)
            | ("sq8", IndexKind::Sq8)
            | ("pq", IndexKind::Pq)
            | ("auto", IndexKind::Auto) => {}
            ("hnsw", IndexKind::Hnsw(p)) | ("hnsw_pq", IndexKind::HnswPq(p)) => {
                assert_eq!((p.m, p.ef_construction), (12, 120), "{mode}");
                let ef = if mode == "hnsw" { 90 } else { 70 };
                assert_eq!(p.ef_search, ef, "{mode}");
            }
            ("ivf", IndexKind::Ivf { nlist, nprobe, .. }) => assert_eq!((nlist, nprobe), (8, 3)),
            ("rabitq", IndexKind::Rabitq(c)) => assert_eq!((c.bits, c.rerank, c.seed), (2, 7, 9)),
            other => panic!("unexpected settings {other:?}"),
        }
        drop(db);
        // A default open (no mode named) also uses the recorded mode
        let db = SrvDB::new_with_config(path, types::DatabaseConfig::new(DIM).unwrap()).unwrap();
        assert_eq!(db.current_settings(), expected, "{mode} via default config");
    }
}

fn mismatch<T>(result: Result<T>) -> bool {
    matches!(result, Err(Error::SettingsMismatch(_)))
}

#[test]
fn conflicting_settings_fail_and_change_nothing() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    drop(create(path, "hnsw"));
    let before = std::fs::read(dir.path().join(FILE_NAME)).unwrap();
    let training: Vec<Vec<f32>> = (0..N).map(vector).collect();

    assert!(mismatch(SrvDB::new_scalar_quantized(path, DIM, &training)));
    assert!(mismatch(SrvDB::new_rabitq(
        path,
        DIM,
        RabitqConfig::new(4).unwrap()
    )));
    assert!(mismatch(SrvDB::new_with_hnsw(
        path,
        DIM,
        hnsw::HNSWConfig::new(32, 120, 70)
    )));
    let mut sq8 = types::DatabaseConfig::new(DIM).unwrap();
    sq8.quantization.enabled = true;
    sq8.quantization.mode = types::QuantizationMode::Scalar;
    assert!(mismatch(SrvDB::new_with_config(path, sq8)));
    let other_dim = SrvDB::new_with_config(path, types::DatabaseConfig::new(256).unwrap());
    assert!(matches!(other_dim, Err(Error::DimensionMismatch { .. })));
    assert_eq!(std::fs::read(dir.path().join(FILE_NAME)).unwrap(), before);

    // Search-time knobs are not conflicts: the caller's value is used
    let db = SrvDB::new_with_hnsw(path, DIM, hnsw::HNSWConfig::new(12, 120, 70)).unwrap();
    assert_eq!(db.count(), N as u64);
}

#[test]
fn rabitq_settings_are_derived_and_rebuild_instead_of_conflicting() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    drop(create(path, "rabitq"));
    // ADR 0008: another bit width rebuilds the codes from the vectors;
    // persist() records the new settings
    let mut db = SrvDB::new_rabitq(path, DIM, RabitqConfig::new(4).unwrap()).unwrap();
    db.persist().unwrap();
    drop(db);
    let db = SrvDB::open_existing(path).unwrap();
    assert_eq!(db.rabitq_config().map(|c| c.bits), Some(4));
    drop(db);
    // Another mode over the same vectors is still a conflict
    assert!(mismatch(SrvDB::new_with_hnsw(path, DIM, hnsw_config())));
}

#[test]
fn mode_changes_are_recorded_by_persist() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let mut db = create(path, "flat");
    db.attach_hnsw(hnsw_config()).unwrap();
    db.persist().unwrap();
    drop(db);
    let db = SrvDB::open_existing(path).unwrap();
    let expected = HnswParams {
        m: 12,
        ef_construction: 120,
        ef_search: 70,
    };
    assert_eq!(db.current_settings().kind, IndexKind::Hnsw(expected));
}
