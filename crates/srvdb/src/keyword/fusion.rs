//! Reciprocal Rank Fusion of ranked result lists (ADR 0015)

use std::cmp::Ordering;
use std::collections::HashMap;

/// RRF constant from Cormack et al. (2009)
pub(crate) const RRF_K: f64 = 60.0;

/// Fuse ranked id lists (best first) with `sum 1 / (RRF_K + rank)`, ranks
/// from 1. Returns the `k` best ids with their fused score, best first,
/// ties broken by lower id.
pub(crate) fn rrf(lists: &[Vec<u64>], k: usize) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f64> = HashMap::new();
    for list in lists {
        for (rank, &id) in list.iter().enumerate() {
            *scores.entry(id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }
    let mut fused: Vec<(u64, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    fused.truncate(k);
    fused.into_iter().map(|(id, s)| (id, s as f32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_formula() {
        let fused = rrf(&[vec![1, 2, 3], vec![3, 1]], 10);
        let s = |r: f64| 1.0 / (RRF_K + r);
        let expected = [(1, s(1.0) + s(2.0)), (3, s(3.0) + s(1.0)), (2, s(2.0))];
        assert_eq!(fused.len(), 3);
        for ((id, score), (eid, escore)) in fused.iter().zip(expected) {
            assert_eq!(*id, eid);
            assert!((f64::from(*score) - escore).abs() < 1e-6);
        }
    }

    #[test]
    fn ties_break_by_id_and_k_truncates() {
        assert_eq!(rrf(&[vec![9], vec![4]], 1)[0].0, 4);
    }
}
