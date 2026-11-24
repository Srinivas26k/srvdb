//! HNSW recall against exact search on clustered data (issue #28)
use super::*;
use std::collections::HashSet;

/// SplitMix64 + Box-Muller: deterministic, independent of `rand` versions
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// Seeded Gaussian clusters, like the bench harness's synthetic dataset
fn clustered(dim: usize, n: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gauss()).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centres[(rng.next_u64() % clusters as u64) as usize];
            c.iter().map(|x| x + 0.6 * rng.gauss()).collect()
        })
        .collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (na * nb)
}

#[test]
fn recall_at_10_on_clustered_data_matches_exact_search() {
    let (dim, n, nq, k) = (64, 2000, 50, 10);
    let all = clustered(dim, n + nq, 20, 28);
    let (base, queries) = all.split_at(n);
    let index = HNSWIndex::new(HNSWConfig::new(16, 200, 100));
    let build = |a: u64, b: u64| cosine_distance(&base[a as usize], &base[b as usize]);
    for id in 0..n as u64 {
        index.insert(id, &build).unwrap();
    }

    let mut hits = 0;
    for q in queries {
        let mut exact: Vec<(usize, f32)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_distance(q, v)))
            .collect();
        exact.sort_by(|a, b| a.1.total_cmp(&b.1));
        let truth: HashSet<u64> = exact[..k].iter().map(|&(i, _)| i as u64).collect();
        let dist = |_: u64, b: u64| cosine_distance(q, &base[b as usize]);
        let found = index.search(0, k, &dist).unwrap();
        hits += found.iter().filter(|(id, _)| truth.contains(id)).count();
    }
    let recall = hits as f64 / (nq * k) as f64;
    assert!(recall >= 0.95, "HNSW recall@10 {recall:.3} < 0.95");
}

#[test]
fn every_node_is_reachable_from_the_entry_point() {
    let (dim, n) = (64, 2000);
    let base = clustered(dim, n, 20, 28);
    let index = HNSWIndex::new(HNSWConfig::new(16, 200, 100));
    let build = |a: u64, b: u64| cosine_distance(&base[a as usize], &base[b as usize]);
    for id in 0..n as u64 {
        index.insert(id, &build).unwrap();
    }
    let nodes = index.nodes.read();
    let entry = index.entry_point.read().unwrap();
    let mut seen = HashSet::from([entry]);
    let mut stack = vec![entry];
    while let Some(id) = stack.pop() {
        for &nb in &nodes[&id].neighbors[0] {
            if seen.insert(nb) {
                stack.push(nb);
            }
        }
    }
    assert_eq!(
        seen.len(),
        n,
        "layer 0 nodes reachable from the entry point"
    );
}

/// The filtered graph walk itself (no exact fallback), issue #25:
/// recall@10 against exact filtered search at 1%, 10% and 50% matching
#[test]
fn filtered_walk_recall_at_10_matches_exact_filtered_search() {
    let (dim, n, nq, k) = (64, 2000, 30, 10);
    let all = clustered(dim, n + nq, 20, 25);
    let (base, queries) = all.split_at(n);
    let index = HNSWIndex::new(HNSWConfig::new(16, 200, 100));
    let build = |a: u64, b: u64| cosine_distance(&base[a as usize], &base[b as usize]);
    for id in 0..n as u64 {
        index.insert(id, &build).unwrap();
    }
    for pct in [1u64, 10, 50] {
        // Hashing the id spreads the filter across clusters
        let accept = |id: u64| id.wrapping_mul(0x9E37_79B9_7F4A_7C15) % 100 < pct;
        let mut hits = 0;
        for q in queries {
            let mut exact: Vec<(u64, f32)> = (0..n as u64)
                .filter(|&id| accept(id))
                .map(|id| (id, cosine_distance(q, &base[id as usize])))
                .collect();
            exact.sort_by(|a, b| a.1.total_cmp(&b.1));
            let truth: HashSet<u64> = exact[..k].iter().map(|&(id, _)| id).collect();
            let dist = |_: u64, b: u64| cosine_distance(q, &base[b as usize]);
            let found = index.search_accepted(0, k, &dist, &accept).unwrap();
            assert_eq!(found.len(), k, "fewer than k at {pct}%");
            assert!(found.iter().all(|&(id, _)| accept(id)));
            hits += found.iter().filter(|(id, _)| truth.contains(id)).count();
        }
        let recall = hits as f64 / (nq * k) as f64;
        assert!(
            recall >= 0.95,
            "filtered walk recall@10 at {pct}%: {recall:.3}"
        );
    }
}
