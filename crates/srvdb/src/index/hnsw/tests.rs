//! HNSW unit tests
use super::*;

#[test]
fn test_random_level_distribution() {
    let config = HNSWConfig::default();
    let index = HNSWIndex::new(config);

    let mut level_counts = [0; 10];
    for _ in 0..1000 {
        let level = index.random_level();
        if level < 10 {
            level_counts[level] += 1;
        }
    }

    // Level 0 should have the most nodes (roughly 50%)
    assert!(level_counts[0] > 400);

    // Higher levels should have exponentially fewer nodes
    assert!(level_counts[1] < level_counts[0]);
    assert!(level_counts[2] < level_counts[1]);
}

#[test]
fn test_hnsw_insert_and_search() {
    let config = HNSWConfig::new(16, 200, 50);
    let index = HNSWIndex::new(config);

    // Simple Euclidean distance for testing
    let vectors: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ];

    // Mock distance function
    let distance_fn = |a: u64, b: u64| -> f32 {
        let va = &vectors[a as usize];
        let vb = &vectors[b as usize];
        ((va[0] - vb[0]).powi(2) + (va[1] - vb[1]).powi(2)).sqrt()
    };

    // Insert vectors
    for i in 0..vectors.len() {
        index.insert(i as u64, &distance_fn).unwrap();
    }

    assert_eq!(index.len(), 5);

    // Search for nearest to vector 0
    let results = index.search(0, 3, &distance_fn).unwrap();
    assert_eq!(results.len(), 3);

    // First result should be itself (distance 0)
    assert_eq!(results[0].0, 0);
    assert!(results[0].1 < 0.001);

    // All results should be reasonably close (HNSW is approximate)
    for (_, dist) in &results {
        assert!(*dist < 3.0); // Max distance in our small test set
    }
}

#[test]
fn test_hnsw_persistence() {
    let config = HNSWConfig::new(16, 200, 50);
    let index = HNSWIndex::new(config);

    // Insert some nodes (using dummy distance function)
    let distance_fn = |a: u64, b: u64| -> f32 { ((a as f32) - (b as f32)).abs() };

    for i in 0..10 {
        index.insert(i, &distance_fn).unwrap();
    }

    // Serialize
    let bytes = index.to_bytes().unwrap();

    // Deserialize
    let loaded = HNSWIndex::from_bytes(&bytes).unwrap();

    assert_eq!(loaded.len(), 10);
    assert_eq!(*loaded.entry_point.read(), *index.entry_point.read());
    assert_eq!(*loaded.max_level.read(), *index.max_level.read());
}

/// Many threads inserting into one graph (as IVF partition filling does)
/// must not panic and must keep almost every node reachable. Parallel
/// builds see slightly stale neighbourhoods, so a few tenths of a percent
/// of nodes can lose all incoming links (sequential builds lose none);
/// heuristic neighbour selection (planned for 0.4) addresses that.
#[test]
fn concurrent_inserts_keep_graph_connected() {
    use rayon::prelude::*;
    use std::collections::HashSet;

    let n = 2000u64;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..64).map(|_| rand::random::<f32>() - 0.5).collect())
        .collect();
    let cosine_distance = |a: u64, b: u64| -> f32 {
        let (va, vb) = (&vectors[a as usize], &vectors[b as usize]);
        let dot: f32 = va.iter().zip(vb).map(|(x, y)| x * y).sum();
        let na: f32 = va.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = vb.iter().map(|x| x * x).sum::<f32>().sqrt();
        1.0 - dot / (na * nb)
    };

    let index = HNSWIndex::new(HNSWConfig::default());
    (0..n)
        .into_par_iter()
        .for_each(|id| index.insert(id, &cosine_distance).unwrap());
    assert_eq!(index.len(), n as usize);

    // Reachability from the entry point on layer 0
    let reachable = {
        let nodes = index.nodes.read();
        let entry = index.entry_point.read().unwrap();
        let mut seen = HashSet::new();
        let mut stack = vec![entry];
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                stack.extend(nodes[&id].neighbors[0].iter().copied());
            }
        }
        seen.len()
    };
    assert!(
        reachable as f32 >= 0.99 * n as f32,
        "reachable {reachable}/{n}"
    );

    let found = (0..n)
        .filter(|&q| index.search(q, 1, &cosine_distance).unwrap()[0].0 == q)
        .count();
    assert!(found as f32 >= 0.99 * n as f32, "self-recall {found}/{n}");
}
