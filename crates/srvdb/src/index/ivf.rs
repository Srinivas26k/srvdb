//! # Inverted File (IVF) Clustering
//!
//! Partitions the vector space into Voronoi cells to accelerate search
//! by restricting the search scope to relevant partitions.

use anyhow::Result;
use parking_lot::RwLock;
use rand::RngExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::hnsw::{HNSWConfig, HNSWIndex};
use crate::Vector;

/// IVF Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFConfig {
    /// Number of partitions (centroids)
    pub nlist: usize,
    /// Number of partitions to probe during search
    pub nprobe: usize,
    /// Maximum iterations for K-Means training
    pub max_iterations: usize,
    /// Convergence tolerance for K-Means
    pub tolerance: f32,
    /// Configuration for the internal HNSW graphs
    pub hnsw_config: HNSWConfig,
}

impl Default for IVFConfig {
    fn default() -> Self {
        Self {
            nlist: 100, // Good for ~1M vectors
            nprobe: 10,
            max_iterations: 20,
            tolerance: 0.001,
            hnsw_config: HNSWConfig::default(),
        }
    }
}

/// The IVF Index structure
/// Manages centroids and the partitioned HNSW graphs
pub struct IVFIndex {
    pub config: IVFConfig,
    /// Centroids (cluster centers) - size of `nlist`
    pub centroids: Vec<Vector>,
    /// The partitions themselves. Each is an independent HNSW graph.
    /// We use RwLock to allow concurrent searching/updating of different partitions.
    pub partitions: Vec<Arc<RwLock<HNSWIndex>>>,
}

impl IVFIndex {
    /// Create a new empty IVF index
    pub fn new(config: IVFConfig) -> Self {
        let mut partitions = Vec::with_capacity(config.nlist);
        for _ in 0..config.nlist {
            partitions.push(Arc::new(RwLock::new(HNSWIndex::new(
                config.hnsw_config.clone(),
            ))));
        }

        Self {
            config,
            centroids: Vec::new(),
            partitions,
        }
    }

    /// Train the centroids using K-Means clustering
    ///
    /// This is a "Course Quantization" step.
    /// Returns the centroids and assigns the training vectors to partitions (though assignment isn't stored here).
    pub fn train(&mut self, training_data: &[Vector]) -> Result<()> {
        if training_data.is_empty() {
            return Ok(());
        }

        let k = self.config.nlist;
        let dim = training_data[0].dim();

        log::debug!(
            "IVF: Training {} centroids on {} vectors...",
            k,
            training_data.len()
        );

        // Spherical k-means: search assigns by cosine, so training clusters
        // unit-length directions and each centroid is the normalised mean
        // direction of its members.
        let units: Vec<Vec<f32>> = training_data.iter().map(|v| unit(&v.data)).collect();

        let mut rng = rand::rng();
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|_| units[rng.random_range(0..units.len())].clone())
            .collect();

        for iter in 0..self.config.max_iterations {
            let assignment: Vec<usize> = units.par_iter().map(|u| nearest(u, &centroids)).collect();

            let mut sums = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];
            for (u, &cluster) in units.iter().zip(&assignment) {
                counts[cluster] += 1;
                for (s, x) in sums[cluster].iter_mut().zip(u) {
                    *s += x;
                }
            }

            let mut shift = 0.0;
            for (i, sum) in sums.into_iter().enumerate() {
                let next = if counts[i] > 0 {
                    unit(&sum)
                } else {
                    // Re-seed an empty cluster so it is not lost
                    units[rng.random_range(0..units.len())].clone()
                };
                shift += crate::search::euclidean_distance(&centroids[i], &next);
                centroids[i] = next;
            }

            log::trace!(
                "IVF k-means iteration {}: centroid shift {:.4}",
                iter + 1,
                shift
            );
            if shift < self.config.tolerance {
                log::debug!("IVF k-means converged");
                break;
            }
        }

        self.centroids = centroids.into_iter().map(Vector::new).collect();

        // New centroids invalidate existing assignments; start with empty
        // partitions so retraining does not duplicate vectors.
        self.partitions = (0..k)
            .map(|_| Arc::new(RwLock::new(HNSWIndex::new(self.config.hnsw_config.clone()))))
            .collect();
        Ok(())
    }

    /// The partition whose centroid is most similar (cosine) to `vector`;
    /// the same rule training uses.
    pub fn find_partition(&self, vector: &Vector) -> usize {
        let centroids: Vec<&[f32]> = self.centroids.iter().map(|c| c.data.as_slice()).collect();
        nearest(&vector.data, &centroids)
    }

    /// Add a vector to the index.
    ///
    /// Note: This does NOT add it to global storage (vectors.bin),
    /// it only adds the ID to the appropriate partition graph.
    pub fn add(
        &self,
        id: u64,
        vector: &Vector,
        distance_fn: &impl Fn(u64, u64) -> f32,
    ) -> Result<()> {
        if self.centroids.is_empty() {
            anyhow::bail!("IVF Index not trained");
        }

        let partition_id = self.find_partition(vector);

        let partition = &self.partitions[partition_id];
        partition.read().insert(id, distance_fn)?; // READ lock for HNSW, HNSW handles internal Write locks?
                                                   // Wait, HNSW insert needs to mutate the graph structure.
                                                   // HNSWIndex uses RwLocks internally for its nodes map.
                                                   // HNSWIndex::insert takes &self (immutable reference).
                                                   // Check hnsw.rs line 181: pub fn insert<F>(&self, ...
                                                   // Yes, HNSWIndex is thread-safe with internal locking.

        Ok(())
    }
}

/// `v` scaled to unit length (zero vectors are left as they are)
fn unit(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Index of the centroid with the highest cosine similarity to `v`
fn nearest<C: AsRef<[f32]>>(v: &[f32], centroids: &[C]) -> usize {
    centroids
        .iter()
        .map(|c| crate::search::cosine_similarity(v, c.as_ref()))
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Search assigns vectors to partitions by cosine, so a centroid must be
    /// the average direction of its members, not the average vector (which
    /// long vectors pull towards themselves)
    #[test]
    fn centroids_are_average_directions() {
        let mut data = Vec::new();
        for _ in 0..50 {
            data.push(Vector::new(vec![100.0, 50.0, 0.0, 0.0])); // long, +y
            data.push(Vector::new(vec![1.0, -0.5, 0.0, 0.0])); // short, -y
        }
        let mut ivf = IVFIndex::new(IVFConfig {
            nlist: 1,
            ..Default::default()
        });
        ivf.train(&data).unwrap();
        let c = &ivf.centroids[0].data;
        let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Equal numbers of each direction: the average direction is +x
        assert!((c[0] / norm - 1.0).abs() < 1e-4, "centroid {c:?}");
        assert!((c[1] / norm).abs() < 1e-4, "centroid {c:?}");
    }
}
