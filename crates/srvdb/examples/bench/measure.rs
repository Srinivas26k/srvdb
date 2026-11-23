//! Measurements: exact ground truth, recall, latency percentiles, peak RSS.

use rayon::prelude::*;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Exact top-k by cosine similarity (SrvDB's metric), by brute force.
/// Returns dataset indices per query.
pub fn ground_truth(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries
        .par_iter()
        .map(|q| {
            let mut scored: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine(q, v), i))
                .collect();
            let k = k.min(scored.len());
            scored.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.0.total_cmp(&a.0));
            scored.truncate(k);
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            scored.into_iter().map(|(_, i)| i).collect()
        })
        .collect()
}

/// Mean fraction of the exact top-k found in the approximate top-k.
pub fn recall(truth: &[Vec<usize>], found: &[Vec<usize>]) -> f64 {
    let total: f64 = truth
        .iter()
        .zip(found)
        .map(|(t, f)| {
            if t.is_empty() {
                return 1.0;
            }
            let hits = f.iter().filter(|i| t.contains(i)).count();
            hits as f64 / t.len() as f64
        })
        .sum();
    total / truth.len().max(1) as f64
}

/// Latency percentile (nearest-rank) in milliseconds.
pub fn percentile_ms(sorted_secs: &[f64], p: f64) -> f64 {
    if sorted_secs.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * sorted_secs.len() as f64).ceil() as usize;
    sorted_secs[rank.clamp(1, sorted_secs.len()) - 1] * 1000.0
}

/// Peak resident set size of this process in MiB (Linux `VmHWM`).
/// `None` on platforms without /proc.
pub fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024.0)
}
