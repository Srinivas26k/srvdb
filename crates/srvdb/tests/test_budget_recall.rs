//! Recall against memory budget (ADR 0018), next to the fixed SQ8, PQ and
//! RaBitQ modes. Ordering only: no recall number is asserted or published
//! (AGENTS.md section 11). Slow in debug builds; CI's release job runs it:
//! `cargo test --release --test test_budget_recall -- --ignored`

use srvdb::{CreateOptions, IndexChoice, SrvDB, Vector, VectorEngine};
use std::collections::HashSet;
use tempfile::TempDir;

const DIM: usize = 256;
const N: u64 = 12_000;
const K: usize = 10;
const QUERIES: u64 = 40;

fn rng(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

/// Points around 64 random centres, so neighbours are not all equidistant
fn data() -> Vec<Vec<f32>> {
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    let centres: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..DIM).map(|_| rng(&mut s)).collect())
        .collect();
    (0..N + QUERIES)
        .map(|i| {
            let c = &centres[(i % 64) as usize];
            c.iter().map(|x| x + 0.3 * rng(&mut s)).collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn exact(points: &[Vec<f32>], query: &[f32]) -> HashSet<u64> {
    let mut scored: Vec<(f32, u64)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (cosine(p, query), i as u64))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.iter().take(K).map(|&(_, i)| i).collect()
}

/// Mean recall@K of `options` on the shared data, and the mode it ran in
fn recall(dir: &TempDir, name: &str, options: &CreateOptions, all: &[Vec<f32>]) -> (f64, String) {
    let path = dir.path().join(name);
    let mut db = SrvDB::create(path.to_str().unwrap(), DIM, options).unwrap();
    let points = &all[..N as usize];
    let vectors: Vec<Vector> = points.iter().cloned().map(Vector::new).collect();
    db.add_batch(&vectors, &vec!["{}".to_string(); vectors.len()])
        .unwrap();
    let mut hits = 0;
    for query in &all[N as usize..] {
        let truth = exact(points, query);
        let found = db.search(&Vector::new(query.clone()), K).unwrap();
        hits += found.iter().filter(|r| truth.contains(&r.id)).count();
    }
    let r = db.memory_report();
    let mode = format!(
        "{}{}",
        r.mode,
        r.bits.map_or(String::new(), |b| b.to_string())
    );
    (hits as f64 / (K as u64 * QUERIES) as f64, mode)
}

fn with(index: IndexChoice, name: &str, value: u64) -> CreateOptions {
    let mut options = CreateOptions::new(index);
    options.set(name, value).unwrap();
    options
}

#[test]
#[ignore = "slow in debug builds; run in the release job"]
fn recall_does_not_rise_as_the_budget_falls() {
    let dir = TempDir::new().unwrap();
    let all = data();
    let training = all[..2000].to_vec();

    // Budgets chosen so each lands on its own rung at N records
    let mut by_budget = Vec::new();
    for (mb, rung) in [(16, "flat"), (3, "rabitq4"), (2, "rabitq2"), (1, "rabitq1")] {
        let options = with(IndexChoice::Auto, "memory_mb", mb);
        let (r, mode) = recall(&dir, &format!("auto{mb}"), &options, &all);
        assert_eq!(mode, rung, "memory_mb = {mb}");
        by_budget.push((mode, r));
    }
    // More memory never gives worse recall (small slack for ties)
    for pair in by_budget.windows(2) {
        assert!(pair[0].1 + 0.02 >= pair[1].1, "{by_budget:?}");
    }
    assert_eq!(by_budget[0].1, 1.0, "flat is exact");

    // Each RaBitQ rung matches the fixed mode it chose (same code path)
    for (mode, r) in &by_budget[1..] {
        let bits: u64 = mode.trim_start_matches("rabitq").parse().unwrap();
        let options = with(IndexChoice::Rabitq, "bits", bits);
        let (fixed, _) = recall(&dir, &format!("fixed{bits}"), &options, &all);
        assert_eq!(*r, fixed, "{mode}");
    }

    // Fixed SQ8 and PQ: never above flat
    let mut sq8 = CreateOptions::new(IndexChoice::Sq8);
    sq8.set_training(training.clone()).unwrap();
    let mut pq = CreateOptions::new(IndexChoice::Pq);
    pq.set_training(training).unwrap();
    for (name, options) in [("sq8", sq8), ("pq", pq)] {
        let (r, _) = recall(&dir, name, &options, &all);
        assert!(r <= by_budget[0].1, "{name}");
    }
}
