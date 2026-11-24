//! Certification rules on hand-made intervals, and interval coverage of
//! real estimates

use super::*;
use crate::quant::rabitq::fastscan::{FastScanCodes, Kernel, QueryLut};
use crate::quant::rabitq::RabitqQuantizer;

fn c(id: u64, lo: f32, hi: f32) -> Candidate {
    Candidate { id, lo, hi }
}

#[test]
fn disjoint_intervals_are_all_certified_or_pruned() {
    let all = vec![
        c(0, 0.9, 0.95),
        c(1, 0.8, 0.85),
        c(2, 0.1, 0.2),
        c(3, 0.0, 0.05),
    ];
    let s = split(all, 2);
    assert_eq!(s.certified.len(), 2);
    assert!(s.uncertain.is_empty());
}

#[test]
fn overlapping_intervals_are_uncertain() {
    let all = vec![
        c(0, 0.9, 0.95),
        c(1, 0.7, 0.85),
        c(2, 0.75, 0.8),
        c(3, 0.0, 0.05),
    ];
    let s = split(all, 2);
    assert_eq!(s.certified, vec![0]);
    let mut u = s.uncertain.clone();
    u.sort();
    assert_eq!(u, vec![1, 2]);
}

#[test]
fn ties_never_certify_more_than_k() {
    let all = (0..5).map(|i| c(i, 0.5, 0.5)).collect();
    let s = split(all, 3);
    assert_eq!(s.certified.len(), 3);
    assert_eq!(s.uncertain.len(), 2);
}

#[test]
fn epsilon_matches_union_bound() {
    assert!(epsilon(10) < epsilon(1_000_000));
    let e = epsilon(1000) as f64;
    let fail = 1000.0 * 2.0 * (-e * e / 2.0).exp();
    assert!((fail - (1.0 - CERTIFIED_CONFIDENCE)).abs() < 1e-4);
}

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 30) as f32) - 1.0
}

fn norm(x: &[f32]) -> f32 {
    x.iter().map(|y| y * y).sum::<f32>().sqrt()
}

/// Every estimate lies in its interval (correlated pairs, so cosines are
/// spread out and the sqrt(1 - s^2) shrink is exercised)
#[test]
fn intervals_cover_true_cosines() {
    for bits in [1u8, 2, 4] {
        for dim in [33usize, 96] {
            let q = RabitqQuantizer::new(dim, bits, 11);
            let mut seed = 3u64 + bits as u64;
            let data: Vec<Vec<f32>> = (0..300)
                .map(|_| (0..dim).map(|_| rng(&mut seed)).collect())
                .collect();
            let mut records = Vec::new();
            for v in &data {
                q.encode_into(v, &mut records);
            }
            let mut codes = FastScanCodes::with_widths(dim, bits);
            codes.extend(&records, q.record_size()).unwrap();
            for (qi, base) in data.iter().take(6).enumerate() {
                let query: Vec<f32> = base
                    .iter()
                    .map(|x| x + 0.3 * qi as f32 * rng(&mut seed))
                    .collect();
                let lut = QueryLut::new(q.prepare(&query).tables());
                let b = Bounds {
                    eps: epsilon(data.len()),
                    lut_error: lut.max_error(),
                };
                let est = codes.scores(&lut, Kernel::detect());
                for (i, v) in data.iter().enumerate() {
                    let dot: f32 = v.iter().zip(&query).map(|(a, b)| a * b).sum();
                    let truth = dot / (norm(v) * norm(&query));
                    let h = b.half_width(est[i], codes.factor(i), codes.width(i).unwrap());
                    let err = (est[i] - truth).abs();
                    assert!(err <= h, "{bits}b d{dim} i{i}: error {err} > {h}");
                }
            }
        }
    }
}
