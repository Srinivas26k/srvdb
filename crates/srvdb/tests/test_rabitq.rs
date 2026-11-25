//! RaBitQ mode (ADR 0008): recall against exact search, reopen, damage,
//! compaction and validation of the settings.

use srvdb::{Error, RabitqConfig, SrvDB, Vector, VectorEngine};
use std::collections::HashSet;
use std::path::Path;
use tempfile::TempDir;

const DIM: usize = 128;
const K: usize = 10;

/// Seeded clustered data: `clusters` Gaussian-ish blobs around random
/// centres; queries are fresh draws from the same blobs
struct Data {
    base: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

fn rng(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

fn clustered(n: usize, queries: usize, clusters: usize, seed: u64) -> Data {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..DIM).map(|_| rng(&mut s)).collect())
        .collect();
    let mut draw = |i: usize| -> Vec<f32> {
        let c = &centres[i % clusters];
        c.iter().map(|x| x + 0.35 * rng(&mut s)).collect()
    };
    Data {
        base: (0..n).map(&mut draw).collect(),
        queries: (n..n + queries).map(&mut draw).collect(),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (n(a) * n(b))
}

fn exact(base: &[Vec<f32>], q: &[f32]) -> HashSet<u64> {
    let mut all: Vec<(u64, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, cosine(q, v)))
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    all.into_iter().take(K).map(|(i, _)| i).collect()
}

fn open(dir: &Path, config: RabitqConfig) -> srvdb::Result<SrvDB> {
    SrvDB::new_rabitq(dir.to_str().unwrap(), DIM, config)
}

fn fill(db: &mut SrvDB, base: &[Vec<f32>]) {
    for chunk in base.chunks(1000) {
        let v: Vec<Vector> = chunk.iter().map(|x| Vector::new(x.clone())).collect();
        let m: Vec<String> = (0..chunk.len()).map(|_| "{}".to_string()).collect();
        db.add_batch(&v, &m).unwrap();
    }
    db.persist().unwrap();
}

fn recall(db: &SrvDB, data: &Data) -> f32 {
    let hits: usize = data
        .queries
        .iter()
        .map(|q| {
            let truth = exact(&data.base, q);
            let got = db.search(&Vector::new(q.clone()), K).unwrap();
            got.iter().filter(|r| truth.contains(&r.id)).count()
        })
        .sum();
    hits as f32 / (data.queries.len() * K) as f32
}

/// Recall@10 for every bit width without rerank and with the default
/// rerank factor. The clusters are tight (dozens to hundreds of nearly
/// equidistant neighbours), a hard case for low-bit codes.
fn check_recall(n: usize, floors: [(u8, f32, f32); 3]) {
    let data = clustered(n, 40, 25, 7);
    for (bits, no_rerank, rerank) in floors {
        let dir = TempDir::new().unwrap();
        let config = RabitqConfig::new(bits).unwrap();
        let mut db = open(dir.path(), config.with_rerank(0)).unwrap();
        fill(&mut db, &data.base);
        let raw = recall(&db, &data);
        db.set_rabitq_rerank(config.rerank).unwrap();
        let reranked = recall(&db, &data);
        assert!(raw >= no_rerank, "{bits} bits without rerank: {raw}");
        assert!(reranked >= rerank, "{bits} bits with rerank: {reranked}");
    }
}

#[test]
fn recall_against_exact_search() {
    check_recall(2000, [(1, 0.25, 0.95), (2, 0.38, 0.95), (4, 0.70, 0.95)]);
}

#[test]
#[ignore = "slow in debug builds; run in the release job"]
fn recall_against_exact_search_20k() {
    check_recall(20_000, [(1, 0.05, 0.78), (2, 0.12, 0.80), (4, 0.58, 0.97)]);
}

fn top_ids(db: &SrvDB, q: &[f32]) -> Vec<u64> {
    let r = db.search(&Vector::new(q.to_vec()), K).unwrap();
    r.iter().map(|r| r.id).collect()
}

#[test]
fn reopen_and_damaged_codes_give_the_same_results() {
    let data = clustered(1500, 5, 10, 3);
    let dir = TempDir::new().unwrap();
    let config = RabitqConfig::new(2).unwrap();
    let mut db = open(dir.path(), config).unwrap();
    fill(&mut db, &data.base);
    let want: Vec<Vec<u64>> = data.queries.iter().map(|q| top_ids(&db, q)).collect();
    drop(db);

    let db = open(dir.path(), config).unwrap();
    assert_eq!(db.count(), 1500);
    let got: Vec<Vec<u64>> = data.queries.iter().map(|q| top_ids(&db, q)).collect();
    assert_eq!(got, want, "reopened");
    drop(db);

    // Damaged codes are derived data: detected and rebuilt, same answers
    let codes = dir.path().join("rabitq_codes.bin");
    let mut bytes = std::fs::read(&codes).unwrap();
    bytes[24 + 500] ^= 0xFF;
    std::fs::write(&codes, &bytes).unwrap();
    let db = open(dir.path(), config).unwrap();
    let got: Vec<Vec<u64>> = data.queries.iter().map(|q| top_ids(&db, q)).collect();
    assert_eq!(got, want, "after rebuilding damaged codes");
    drop(db);

    // Another bit width rebuilds the codes too
    let db = open(dir.path(), RabitqConfig::new(4).unwrap()).unwrap();
    assert_eq!(db.rabitq_config().unwrap().bits, 4);
    assert_eq!(top_ids(&db, &data.base[42])[0], 42);
}

#[test]
fn unpersisted_vectors_are_dropped_and_deletes_and_compaction_work() {
    let data = clustered(600, 0, 5, 9);
    let dir = TempDir::new().unwrap();
    let config = RabitqConfig::new(1).unwrap();
    let mut db = open(dir.path(), config).unwrap();
    fill(&mut db, &data.base[..500]);
    let extra: Vec<Vector> = data.base[500..]
        .iter()
        .map(|x| Vector::new(x.clone()))
        .collect();
    let metas = vec!["{}".to_string(); extra.len()];
    db.add_batch(&extra, &metas).unwrap();
    assert_eq!(
        top_ids(&db, &data.base[550])[0],
        550,
        "searchable before persist"
    );
    drop(db); // no persist: the last 100 are discarded

    let mut db = open(dir.path(), config).unwrap();
    assert_eq!(db.count(), 500);
    assert!(top_ids(&db, &data.base[550]).iter().all(|&id| id < 500));
    assert!(db.delete(7).unwrap());
    assert!(!top_ids(&db, &data.base[7]).contains(&7));
    assert_eq!(db.compact().unwrap(), 1);
    // Old id 8 is now id 7; codes were rebuilt for the new ids
    assert_eq!(top_ids(&db, &data.base[8])[0], 7);
}

#[test]
fn bad_settings_are_rejected() {
    for bits in [0u8, 3, 5, 8, 16] {
        let e = RabitqConfig::new(bits).unwrap_err();
        assert!(matches!(e, Error::InvalidInput(_)), "{bits}: {e:?}");
    }
    let dir = TempDir::new().unwrap();
    let mut config = RabitqConfig::new(4).unwrap();
    config.bits = 3;
    let e = open(dir.path(), config).err().unwrap();
    assert!(matches!(e, Error::InvalidInput(_)), "{e:?}");
    let e = SrvDB::new_rabitq(
        dir.path().to_str().unwrap(),
        0,
        RabitqConfig::new(1).unwrap(),
    );
    assert!(matches!(e.err().unwrap(), Error::InvalidInput(_)));

    let mut db = open(dir.path(), RabitqConfig::new(4).unwrap()).unwrap();
    let e = db.set_rabitq_rerank(usize::MAX).unwrap_err();
    assert!(matches!(e, Error::InvalidInput(_)), "{e:?}");
    let e = db.search(&Vector::new(vec![0.5; DIM + 1]), K).unwrap_err();
    assert!(matches!(e, Error::DimensionMismatch { .. }), "{e:?}");
}
