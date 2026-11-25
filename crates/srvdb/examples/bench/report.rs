//! Environment capture and result formatting (JSON + table).

use serde_json::{json, Value};
use std::process::Command;

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn os_description() -> String {
    let pretty = std::fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                .map(|v| v.trim_matches('"').to_string())
        });
    let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .ok()
        .map(|k| k.trim().to_string());
    let mut parts = vec![format!(
        "{}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )];
    parts.extend(pretty);
    parts.extend(kernel.map(|k| format!("kernel {k}")));
    parts.join(", ")
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn commit() -> String {
    match git(&["rev-parse", "HEAD"]) {
        Some(hash) => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|s| !s.is_empty());
            if dirty {
                format!("{hash}-dirty")
            } else {
                hash
            }
        }
        None => "unknown".into(),
    }
}

/// Describe the machine and build that produced the numbers.
pub fn environment(command: &str) -> Value {
    json!({
        "cpu_model": cpu_model(),
        "logical_cores": std::thread::available_parallelism().map_or(0, |n| n.get()),
        "os": os_description(),
        "commit": commit(),
        "srvdb_version": env!("CARGO_PKG_VERSION"),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "command": command,
    })
}

fn f(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(f64::NAN)
}

/// Render results as a Markdown table preceded by the environment.
pub fn table(env: &Value, dataset: &Value, results: &[Value], k: usize) -> String {
    let s = |v: &Value, key: &str| {
        v.get(key)
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string()
    };
    let mut out = String::new();
    out.push_str(&format!(
        "CPU:      {} ({} logical cores)\n",
        s(env, "cpu_model"),
        env["logical_cores"]
    ));
    out.push_str(&format!("OS:       {}\n", s(env, "os")));
    out.push_str(&format!(
        "Commit:   {} ({} build)\n",
        s(env, "commit"),
        s(env, "profile")
    ));
    out.push_str(&format!("Dataset:  {}\n", s(dataset, "name")));
    out.push_str(&format!("Command:  {}\n\n", s(env, "command")));
    if dataset.get("filter_pct").is_some_and(|p| !p.is_null()) {
        return out + &filtered_table(results, k);
    }
    out.push_str(&format!(
        "| mode | recall@{k} | QPS | p50 ms | p99 ms | build s | peak RSS MiB |\n"
    ));
    out.push_str("|------|---------:|----:|-------:|-------:|--------:|-------------:|\n");
    for r in results {
        if let Some(err) = r.get("error").and_then(Value::as_str) {
            out.push_str(&format!("| {} | error: {err} | | | | | |\n", s(r, "mode")));
            continue;
        }
        let rss = r
            .get("peak_rss_mib")
            .and_then(Value::as_f64)
            .map_or("n/a".into(), |m| format!("{m:.0}"));
        out.push_str(&format!(
            "| {} | {:.4} | {:.0} | {:.3} | {:.3} | {:.2} | {} |\n",
            s(r, "mode"),
            f(r, "recall_at_k"),
            f(r, "qps"),
            f(r, "p50_ms"),
            f(r, "p99_ms"),
            f(r, "build_secs"),
            rss
        ));
    }
    out
}

/// Filtered runs: filtered search next to post-filtering, per mode.
fn filtered_table(results: &[Value], k: usize) -> String {
    let mut out = format!(
        "| mode | filter % | matching | recall@{k} | QPS | p99 ms | short | \
         post-filter recall@{k} | post-filter QPS | post-filter short |\n"
    );
    out.push_str("|------|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n");
    for r in results {
        let mode = r.get("mode").and_then(Value::as_str).unwrap_or("?");
        if let Some(err) = r.get("error").and_then(Value::as_str) {
            out.push_str(&format!("| {mode} | error: {err} | | | | | | | | |\n"));
            continue;
        }
        let n = |key: &str| r.get(key).and_then(Value::as_u64).unwrap_or(0);
        out.push_str(&format!(
            "| {mode} | {} | {} | {:.4} | {:.0} | {:.3} | {} | {:.4} | {:.0} | {} |\n",
            f(r, "filter_pct"),
            n("matching"),
            f(r, "recall_at_k"),
            f(r, "qps"),
            f(r, "p99_ms"),
            n("short_queries"),
            f(r, "post_filter_recall"),
            f(r, "post_filter_qps"),
            n("post_filter_short_queries"),
        ));
    }
    out
}
