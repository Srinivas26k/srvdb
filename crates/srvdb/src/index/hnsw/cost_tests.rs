//! Measurement behind the filtered-search exact-scan threshold
//! (engine::filter::EXACT_COST_RATIO, ADR 0010, docs/benchmarks.md).
//!
//! Run in release mode; it prints numbers and asserts nothing about speed:
//!
//! ```text
//! cargo test --release -p srvdb filtered_walk_cost -- --ignored --nocapture
//! ```
use super::recall_tests::{clustered, cosine_distance};
use super::*;
use std::cell::Cell;
use std::time::Instant;

#[test]
#[ignore = "measurement; run in release with --ignored --nocapture"]
fn filtered_walk_cost() {
    let (dim, n, nq, k) = (64, 20_000, 50, 10);
    let all = clustered(dim, n + nq, 200, 25);
    let (base, queries) = all.split_at(n);
    let index = HNSWIndex::new(HNSWConfig::new(16, 100, 100));
    let build = |a: u64, b: u64| cosine_distance(&base[a as usize], &base[b as usize]);
    for id in 0..n as u64 {
        index.insert(id, &build).unwrap();
    }
    println!("n={n} dim={dim} queries={nq} M=16 ef=100");
    println!("pct | visits/query | walk us/query | us/visit | exact us/match | ratio");
    for pct in [100u64, 50, 10, 1] {
        let accept = |id: u64| id.wrapping_mul(0x9E37_79B9_7F4A_7C15) % 100 < pct;
        let matches: Vec<u64> = (0..n as u64).filter(|&id| accept(id)).collect();
        let visits = Cell::new(0u64);
        let t = Instant::now();
        for q in queries {
            let dist = |_: u64, b: u64| {
                visits.set(visits.get() + 1);
                cosine_distance(q, &base[b as usize])
            };
            std::hint::black_box(index.search_accepted(0, k, &dist, &accept).unwrap());
        }
        let walk_us = t.elapsed().as_secs_f64() * 1e6 / nq as f64;
        let t = Instant::now();
        for q in queries {
            let mut scored: Vec<(u64, f32)> = matches
                .iter()
                .map(|&id| (id, cosine_distance(q, &base[id as usize])))
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            std::hint::black_box(&scored);
        }
        let exact_us = t.elapsed().as_secs_f64() * 1e6 / (nq * matches.len()) as f64;
        let per_query = visits.get() as f64 / nq as f64;
        // Equal cost when s * n * exact = walk, i.e. s^2 * n = ratio * ef * M
        let s = pct as f64 / 100.0;
        let ratio = s * walk_us / exact_us / (100.0 * 16.0);
        println!(
            "{pct:>3} | {per_query:>12.0} | {walk_us:>13.0} | {:>8.3} | {exact_us:>14.3} | {ratio:>5.1}",
            walk_us / per_query
        );
    }
}
