//! k-means codebook for one product-quantization subspace

use super::product::K;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Codebook for a single sub-quantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    pub centroids: Vec<Vec<f32>>, // K centroids, each d_sub dim
}

/// Normalize a vector to unit length (L2 norm = 1.0)
#[inline]
fn normalize_vector(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

impl Codebook {
    /// Create new codebook with k-means clustering
    pub fn train(vectors: &[Vec<f32>], k: usize, max_iter: usize) -> Result<Self> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot train on empty dataset");
        }

        let d_sub = vectors[0].len();

        // Use vectors as-is (don't normalize sub-vectors)
        // Cosine similarity will be computed via magnitude normalization in distance calculation
        let mut centroids = Self::kmeans_plusplus_init(vectors, k, d_sub);

        for _ in 0..max_iter {
            // Assignment step
            let assignments: Vec<usize> = vectors
                .par_iter()
                .map(|v| Self::nearest_centroid(v, &centroids))
                .collect();

            // Update step
            let mut new_centroids = vec![vec![0.0f32; d_sub]; k];
            let mut counts = vec![0usize; k];

            for (vec, &cluster_id) in vectors.iter().zip(assignments.iter()) {
                for i in 0..d_sub {
                    new_centroids[cluster_id][i] += vec[i];
                }
                counts[cluster_id] += 1;
            }

            // Average to get new centroids
            let mut converged = true;
            for i in 0..k {
                if counts[i] > 0 {
                    for val in new_centroids[i].iter_mut() {
                        *val /= counts[i] as f32;
                    }
                    // Normalize centroids to unit length for proper cosine similarity
                    normalize_vector(&mut new_centroids[i]);

                    // Check convergence
                    if Self::euclidean_distance(&centroids[i], &new_centroids[i]) > 1e-6 {
                        converged = false;
                    }
                } else {
                    // Reinitialize empty clusters
                    let random_idx = rand::random::<usize>() % vectors.len();
                    new_centroids[i] = vectors[random_idx].clone();
                    converged = false;
                }
            }

            centroids = new_centroids;

            if converged {
                break;
            }
        }

        Ok(Self { centroids })
    }

    /// K-means++ initialization for better clustering
    fn kmeans_plusplus_init(vectors: &[Vec<f32>], k: usize, _d_sub: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // Pick first centroid randomly
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx].clone());

        // Pick remaining centroids with probability proportional to distance squared
        for _ in 1..k {
            let distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| Self::euclidean_distance(v, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .map(|d| d * d)
                .collect();

            let total: f32 = distances.iter().sum();

            if total <= 1e-10 {
                // Determine random remaining
                let random_idx = rng.gen_range(0..vectors.len());
                centroids.push(vectors[random_idx].clone());
                continue;
            }

            let mut target = rng.gen::<f32>() * total;

            for (i, &dist) in distances.iter().enumerate() {
                target -= dist;
                if target <= 0.0 {
                    centroids.push(vectors[i].clone());
                    break;
                }
            }
        }

        centroids
    }

    /// Find nearest centroid index using cosine similarity
    #[inline]
    fn nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> usize {
        // Calculate magnitude of input vector
        let vec_magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        if vec_magnitude < 1e-10 {
            return 0; // Default to first centroid if zero vector
        }

        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                // Compute centroid magnitude
                let c_magnitude: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();

                if c_magnitude < 1e-10 {
                    return (i, -1.0); // Very low similarity for zero centroid
                }

                // Compute cosine similarity
                let dot: f32 = vec.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                let cosine = dot / (vec_magnitude * c_magnitude);
                (i, cosine)
            })
            .max_by(|(_, s1), (_, s2)| s1.total_cmp(s2))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Euclidean distance between two vectors
    #[inline]
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Encode a sub-vector to its nearest centroid index
    #[inline]
    pub fn encode(&self, sub_vec: &[f32]) -> u8 {
        // Use sub-vector as-is, distance computation handles normalization
        Self::nearest_centroid(sub_vec, &self.centroids) as u8
    }

    /// Compute distances from query sub-vector to all centroids
    /// Centroids are normalized, query sub-vector is not
    /// Returns NEGATIVE cosine similarity (for minimization-based search)
    #[inline]
    pub fn compute_distances(&self, query_sub: &[f32]) -> Vec<f32> {
        // Calculate magnitude of query sub-vector
        let query_magnitude: f32 = query_sub.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut distances = vec![0.0f32; K];

        // If query magnitude is near zero, return zeros
        if query_magnitude < 1e-10 {
            return distances;
        }

        for (i, centroid) in self.centroids.iter().enumerate() {
            // Compute dot product (centroid is normalized to unit length)
            let dot_product: f32 = query_sub
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| a * b)
                .sum();

            // Divide by query magnitude to get cosine (centroid magnitude is 1.0)
            let cosine = dot_product / query_magnitude;

            // Negate because we minimize distance but want to maximize similarity
            distances[i] = -cosine;
        }
        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_training() {
        let d_sub = 8;
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| {
                let mut v = vec![0.0f32; d_sub];
                for (j, x) in v.iter_mut().enumerate() {
                    *x = (i as f32 / 1000.0) + (j as f32 / 10.0);
                }
                v
            })
            .collect();

        let codebook = Codebook::train(&vectors, 16, 20).unwrap();
        assert_eq!(codebook.centroids.len(), 16);
    }
}
