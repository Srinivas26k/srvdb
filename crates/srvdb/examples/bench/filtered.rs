//! Filtered mode (`--filter-pct P`): every mode searches only the vectors
//! of a seeded random subset holding P% of the base set.
//!
//! Reports recall@k and QPS of `search_filtered` against exact filtered
//! search, plus the same numbers for post-filtering (search k, then drop
//! non-matching results), the only option before issue #25.

use crate::args::Args;
use crate::dataset::{Dataset, SplitMix};
use crate::measure;
use serde_json::{json, Value};
use srvdb::{SrvDB, Vector, VectorEngine};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Dataset positions in the filter: each kept with probability pct/100,
/// independent of the clusters, so the filter cuts across them.
pub fn matching(n: usize, pct: f64, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix::new(seed ^ 0xF117_E5ED);
    (0..n).filter(|_| rng.unit() * 100.0 < pct).collect()
}

/// Run all queries filtered and post-filtered; returns the result fields.
pub fn run(db: &SrvDB, ds: &Dataset, args: &Args, ids: &[u64], pct: f64) -> srvdb::Result<Value> {
    let keep = matching(ds.base.len(), pct, args.seed);
    let allowed: HashSet<u64> = keep.iter().map(|&i| ids[i]).collect();
    let pos: HashMap<u64, usize> = ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let queries: Vec<Vector> = ds.queries.iter().map(|q| Vector::new(q.clone())).collect();
    for q in queries.iter().take(10) {
        db.search_filtered(q, args.k, &allowed)?;
    }

    let to_pos = |ids: Vec<u64>| ids.iter().filter_map(|id| pos.get(id).copied()).collect();
    let mut lat = Vec::with_capacity(queries.len());
    let mut found: Vec<Vec<usize>> = Vec::with_capacity(queries.len());
    let t = Instant::now();
    for q in &queries {
        let s = Instant::now();
        let res = db.search_filtered(q, args.k, &allowed)?;
        lat.push(s.elapsed().as_secs_f64());
        found.push(to_pos(res.into_iter().map(|r| r.id).collect()));
    }
    let secs = t.elapsed().as_secs_f64();

    let mut post: Vec<Vec<usize>> = Vec::with_capacity(queries.len());
    let t = Instant::now();
    for q in &queries {
        let res = db.search(q, args.k)?;
        let ids = res
            .into_iter()
            .map(|r| r.id)
            .filter(|id| allowed.contains(id));
        post.push(to_pos(ids.collect()));
    }
    let post_secs = t.elapsed().as_secs_f64();

    // Exact filtered truth: brute force over the matching subset only
    let subset: Vec<Vec<f32>> = keep.iter().map(|&i| ds.base[i].clone()).collect();
    let truth: Vec<Vec<usize>> = measure::ground_truth(&subset, &ds.queries, args.k)
        .into_iter()
        .map(|t| t.into_iter().map(|j| keep[j]).collect())
        .collect();
    let short = |f: &[Vec<usize>]| f.iter().filter(|r| r.len() < args.k).count();
    lat.sort_by(f64::total_cmp);
    let nq = queries.len() as f64;
    Ok(json!({
        "filter_pct": pct,
        "matching": keep.len(),
        "recall_at_k": measure::recall(&truth, &found),
        "qps": nq / secs.max(f64::EPSILON),
        "p50_ms": measure::percentile_ms(&lat, 50.0),
        "p99_ms": measure::percentile_ms(&lat, 99.0),
        "short_queries": short(&found),
        "post_filter_recall": measure::recall(&truth, &post),
        "post_filter_qps": nq / post_secs.max(f64::EPSILON),
        "post_filter_short_queries": short(&post),
    }))
}
