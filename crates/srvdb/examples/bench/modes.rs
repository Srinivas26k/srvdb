//! Build and query one SrvDB mode, timing each phase.

use crate::args::Args;
use crate::dataset::Dataset;
use crate::measure;
use serde_json::{json, Value};
use srvdb::types::DatabaseConfig;
use srvdb::{HNSWConfig, IVFConfig, IndexMode, SrvDB, Vector, VectorEngine};
use std::time::Instant;

/// Vectors used to train SQ8 / PQ quantizers (and IVF k-means internally).
const TRAIN_LIMIT: usize = 10_000;

fn create(mode: &str, path: &str, ds: &Dataset, args: &Args) -> srvdb::Result<SrvDB> {
    let train = &ds.base[..ds.base.len().min(TRAIN_LIMIT)];
    match mode {
        "flat" | "ivf" => {
            let cfg = DatabaseConfig::new(ds.dim)
                .map_err(|e| srvdb::Error::InvalidInput(e.to_string()))?;
            SrvDB::new_with_config(path, cfg)
        }
        "sq8" => SrvDB::new_scalar_quantized(path, ds.dim, train),
        "pq" => {
            let t: Vec<Vector> = train.iter().map(|v| Vector::new(v.clone())).collect();
            SrvDB::new_quantized(path, &t)
        }
        "hnsw" => {
            let cfg = HNSWConfig {
                ef_search: args.ef_search,
                ..HNSWConfig::default()
            };
            SrvDB::new_with_hnsw(path, ds.dim, cfg)
        }
        other => Err(srvdb::Error::InvalidInput(format!("unknown mode {other}"))),
    }
}

fn build(mode: &str, path: &str, ds: &Dataset, args: &Args) -> srvdb::Result<(SrvDB, Vec<u64>)> {
    let mut db = create(mode, path, ds, args)?;
    let mut ids = Vec::with_capacity(ds.base.len());
    for chunk in ds.base.chunks(1000) {
        let vecs: Vec<Vector> = chunk.iter().map(|v| Vector::new(v.clone())).collect();
        let metas: Vec<String> = (0..chunk.len()).map(|_| "{}".to_string()).collect();
        ids.extend(db.add_batch(&vecs, &metas)?);
    }
    if mode == "ivf" {
        let nlist = args
            .nlist
            .unwrap_or_else(|| ((ds.base.len() as f64).sqrt() as usize).max(1));
        let mut cfg = IVFConfig {
            nlist,
            nprobe: args.nprobe.min(nlist),
            ..IVFConfig::default()
        };
        cfg.hnsw_config.ef_search = args.ef_search;
        db.configure_ivf(cfg)?;
        db.set_mode(IndexMode::Ivf);
        db.train_ivf()?;
    }
    if mode == "hnsw" || mode == "ivf" {
        db.set_ef_search(args.ef_search)?;
    }
    db.persist()?;
    Ok((db, ids))
}

/// Build, query and measure one mode. Returns a JSON result object.
pub fn run(mode: &str, ds: &Dataset, args: &Args, path: &str) -> Result<Value, String> {
    let fail = |stage: &str, e: srvdb::Error| format!("{mode}: {stage} failed: {e}");

    let t0 = Instant::now();
    let (db, ids) = build(mode, path, ds, args).map_err(|e| fail("build", e))?;
    let build_secs = t0.elapsed().as_secs_f64();

    if let Some(pct) = args.filter_pct {
        let mut v =
            crate::filtered::run(&db, ds, args, &ids, pct).map_err(|e| fail("search", e))?;
        v["mode"] = json!(mode);
        v["build_secs"] = json!(build_secs);
        v["peak_rss_mib"] = json!(measure::peak_rss_mib());
        return Ok(v);
    }

    // Map database ids back to dataset positions for recall.
    let pos: std::collections::HashMap<u64, usize> =
        ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();

    let queries: Vec<Vector> = ds.queries.iter().map(|q| Vector::new(q.clone())).collect();
    for q in queries.iter().take(10) {
        db.search(q, args.k).map_err(|e| fail("warm-up", e))?;
    }

    let mut lat = Vec::with_capacity(queries.len());
    let mut found = Vec::with_capacity(queries.len());
    let t1 = Instant::now();
    for q in &queries {
        let s = Instant::now();
        let res = db.search(q, args.k).map_err(|e| fail("search", e))?;
        lat.push(s.elapsed().as_secs_f64());
        found.push(
            res.iter()
                .filter_map(|r| pos.get(&r.id).copied())
                .collect::<Vec<_>>(),
        );
    }
    let search_secs = t1.elapsed().as_secs_f64();
    let peak_rss = measure::peak_rss_mib();
    drop(db);

    let truth = measure::ground_truth(&ds.base, &ds.queries, args.k);
    lat.sort_by(f64::total_cmp);
    Ok(json!({
        "mode": mode,
        "recall_at_k": measure::recall(&truth, &found),
        "qps": queries.len() as f64 / search_secs.max(f64::EPSILON),
        "p50_ms": measure::percentile_ms(&lat, 50.0),
        "p99_ms": measure::percentile_ms(&lat, 99.0),
        "build_secs": build_secs,
        "peak_rss_mib": peak_rss,
    }))
}
