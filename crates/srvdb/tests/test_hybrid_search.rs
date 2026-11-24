//! Hybrid (vector + BM25, Reciprocal Rank Fusion) search (ADR 0015)

use srvdb::{Error, Filter, HNSWConfig, SrvDB, Vector, VectorEngine};
use std::collections::HashSet;
use tempfile::TempDir;

const DIM: usize = 128;
const N: usize = 400;
const WORDS: [&str; 8] = [
    "alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "zeta",
];

fn vector(i: usize) -> Vector {
    Vector::new(
        (0..DIM)
            .map(|j| (((i * 31 + j * 17) % 97) as f32 - 48.0) / 10.0 + (i % 5) as f32)
            .collect(),
    )
}

fn meta(i: usize) -> String {
    let text = format!("{} {} doc{i}", WORDS[i % 8], WORDS[(i / 8) % 8]);
    serde_json::json!({ "text": text, "even": i.is_multiple_of(2) }).to_string()
}

fn fill(db: &mut SrvDB) {
    let vectors: Vec<Vector> = (0..N).map(vector).collect();
    let metas: Vec<String> = (0..N).map(meta).collect();
    db.add_batch(&vectors, &metas).unwrap();
    db.enable_keyword_index("text").unwrap();
}

fn flat(dir: &TempDir) -> SrvDB {
    let path = dir.path().to_str().unwrap();
    let mut db = SrvDB::new_with_config(path, srvdb::DatabaseConfig::new(DIM).unwrap()).unwrap();
    fill(&mut db);
    db
}

fn ids(results: &[srvdb::SearchResult]) -> Vec<u64> {
    results.iter().map(|r| r.id).collect()
}

#[test]
fn fusion_contains_the_best_of_both_lists() {
    let dir = TempDir::new().unwrap();
    let db = flat(&dir);
    let query = vector(123);
    let hybrid = db.search_hybrid(&query, "doc7", 5, None).unwrap();
    let got = ids(&hybrid);
    assert_eq!(got.len(), 5);
    let vector_top = ids(&db.search(&query, 1).unwrap())[0];
    assert!(got.contains(&vector_top), "{got:?}");
    assert!(got.contains(&7), "{got:?}");
    // Scores are RRF scores: an item first in one list scores >= 1/61
    assert!(hybrid[0].score >= 1.0 / 61.0 - 1e-6);
    assert!(hybrid.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(hybrid.iter().all(|r| r.metadata.is_some()));
}

#[test]
fn filter_applies_to_both_sides() {
    let dir = TempDir::new().unwrap();
    let db = flat(&dir);
    let filter = Filter::from_json(r#"{"even": true}"#).unwrap();
    let got = db
        .search_hybrid(&vector(3), "alpha doc3", 10, Some(&filter))
        .unwrap();
    assert_eq!(got.len(), 10);
    assert!(
        got.iter().all(|r| r.id.is_multiple_of(2)),
        "{:?}",
        ids(&got)
    );
}

#[test]
fn hnsw_hybrid_recall_against_flat_hybrid() {
    let flat_dir = TempDir::new().unwrap();
    let exact = flat(&flat_dir);
    let hnsw_dir = TempDir::new().unwrap();
    let path = hnsw_dir.path().to_str().unwrap();
    let mut approx = SrvDB::new_with_hnsw(path, DIM, HNSWConfig::new(16, 200, 100)).unwrap();
    fill(&mut approx);

    let (mut hit, mut total) = (0usize, 0usize);
    for q in (0..N).step_by(20) {
        let text = WORDS[q % 8];
        let want: HashSet<u64> = ids(&exact.search_hybrid(&vector(q), text, 10, None).unwrap())
            .into_iter()
            .collect();
        let got = ids(&approx.search_hybrid(&vector(q), text, 10, None).unwrap());
        hit += got.iter().filter(|id| want.contains(id)).count();
        total += want.len();
    }
    let recall = hit as f64 / total as f64;
    assert!(recall >= 0.9, "hybrid recall {recall}");
}

#[test]
fn invalid_arguments_are_rejected() {
    let dir = TempDir::new().unwrap();
    let db = flat(&dir);
    let short = Vector::new(vec![1.0; 3]);
    assert!(matches!(
        db.search_hybrid(&short, "alpha", 3, None),
        Err(Error::DimensionMismatch { .. })
    ));
    let nan = Vector::new(vec![f32::NAN; DIM]);
    assert!(matches!(
        db.search_hybrid(&nan, "alpha", 3, None),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        db.search_hybrid(&vector(0), "alpha", 0, None),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        db.search_hybrid(&vector(0), "--", 3, None),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn hybrid_needs_the_keyword_index() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let db = SrvDB::new_with_config(path, srvdb::DatabaseConfig::new(DIM).unwrap()).unwrap();
    let err = db.search_hybrid(&vector(0), "alpha", 3, None).unwrap_err();
    assert!(matches!(err, Error::InvalidState(_)), "{err:?}");
}
