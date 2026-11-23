//! Reproducible benchmark harness for SrvDB (issue #21).
//!
//! Measures recall@k against exact brute-force search, QPS, p50/p99
//! latency, build time and peak RSS for each mode (flat, sq8, pq, hnsw,
//! ivf). Each mode runs in its own child process so peak RSS is per mode.
//! See docs/benchmarks.md for how to reproduce published numbers.

mod args;
mod dataset;
mod measure;
mod modes;
mod report;

use serde_json::{json, Value};
use std::process::Command;

fn load(a: &args::Args) -> Result<dataset::Dataset, String> {
    match (&a.base_fvecs, &a.query_fvecs) {
        (Some(b), Some(q)) => dataset::from_fvecs(b, q, a.n, a.queries),
        _ => Ok(dataset::synthetic(a.dim, a.n, a.queries, a.seed)),
    }
}

/// Child: run one mode and print its result as one JSON line on stdout.
fn child(a: &args::Args, mode: &str) -> Result<(), String> {
    let ds = load(a)?;
    let dir = match &a.work_dir {
        Some(w) => {
            std::fs::create_dir_all(w).map_err(|e| format!("work dir: {e}"))?;
            tempfile::tempdir_in(w)
        }
        None => tempfile::tempdir(),
    }
    .map_err(|e| format!("temp dir: {e}"))?;
    let path = dir.path().join(mode);
    let path = path.to_str().ok_or("temp path is not UTF-8")?;
    let result = modes::run(mode, &ds, a, path)?;
    println!("{result}");
    Ok(())
}

fn run_mode_process(raw: &[String], mode: &str) -> Value {
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return json!({"mode": mode, "error": format!("current_exe: {e}")}),
    };
    let out = Command::new(exe).args(raw).args(["--child", mode]).output();
    match out {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let line = stdout.lines().last().unwrap_or("");
            serde_json::from_str(line).unwrap_or_else(
                |e| json!({"mode": mode, "error": format!("bad child output: {e}")}),
            )
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
            json!({"mode": mode, "error": err})
        }
        Err(e) => json!({"mode": mode, "error": format!("spawn: {e}")}),
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let a = match args::parse(&raw) {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}\n");
            }
            eprint!("{}", args::USAGE);
            std::process::exit(2);
        }
    };
    if let Some(mode) = &a.child {
        if let Err(e) = child(&a, mode) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }

    let ds = match load(&a) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let dataset = json!({
        "name": ds.name, "dim": ds.dim, "base": ds.base.len(),
        "queries": ds.queries.len(), "k": a.k, "seed": a.seed,
        "ef_search": a.ef_search, "nprobe": a.nprobe, "nlist": a.nlist,
    });
    drop(ds);

    let command = format!(
        "cargo run --release -p srvdb --example bench -- {}",
        raw.join(" ")
    );
    let env = report::environment(command.trim_end());
    let mut results = Vec::new();
    for mode in &a.modes {
        eprintln!("running {mode} ...");
        results.push(run_mode_process(&raw, mode));
    }

    let doc = json!({"environment": env, "dataset": dataset, "results": results});
    let pretty = serde_json::to_string_pretty(&doc).unwrap_or_else(|_| doc.to_string());
    if let Err(e) = std::fs::write(&a.out, pretty + "\n") {
        eprintln!("error: cannot write {}: {e}", a.out.display());
        std::process::exit(1);
    }
    print!("{}", report::table(&env, &dataset, &results, a.k));
    println!("\nJSON written to {}", a.out.display());
    if results.iter().any(|r| r.get("error").is_some()) {
        std::process::exit(1);
    }
}
