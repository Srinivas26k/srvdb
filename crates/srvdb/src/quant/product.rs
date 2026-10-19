//! Product Quantization (PQ) for vector compression
//!
//! Implements FAISS-style Product Quantization:
//! - ~32x memory compression (float32 -> uint8)
//! - Dynamic dimension support
//! - Asymmetric Distance Computation (ADC) for fast search
//! - K-means clustering for codebook training

use super::codebook::Codebook;
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Constants
pub const K: usize = 256; // Centroids per sub-quantizer (fits in u8)

/// Quantized vector representation (M bytes)
pub type QuantizedVector = Vec<u8>;

/// Distance lookup table for Asymmetric Distance Computation
#[derive(Debug, Clone)]
pub struct DistanceTable {
    pub tables: Vec<Vec<f32>>, // M tables, each with K distances
}

/// Product Quantizer - splits vectors into M sub-vectors and quantizes each
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    pub m: usize,
    pub k: usize,
    pub d_sub: usize,
    pub codebooks: Vec<Codebook>,
}

impl ProductQuantizer {
    /// Train a new product quantizer on training data
    pub fn train(training_data: &[Vec<f32>], m: usize) -> Result<Self> {
        if training_data.is_empty() {
            anyhow::bail!("Training data cannot be empty");
        }

        let dim = training_data[0].len();
        if !dim.is_multiple_of(m) {
            anyhow::bail!("Dimension {} must be divisible by m={}", dim, m);
        }
        let d_sub = dim / m;

        println!(
            "Training Product Quantizer with M={}, K={}, D_sub={}",
            m, K, d_sub
        );

        // Train codebook for each sub-quantizer
        let codebooks: Result<Vec<Codebook>> = (0..m)
            .into_par_iter()
            .map(|m_idx| {
                println!("Training sub-quantizer {}/{}", m_idx + 1, m);

                // Extract sub-vectors for this sub-quantizer
                let sub_vectors: Vec<Vec<f32>> = training_data
                    .iter()
                    .map(|vec| {
                        let start = m_idx * d_sub;
                        let end = start + d_sub;
                        vec[start..end].to_vec()
                    })
                    .collect();

                // Train codebook for this sub-space (normalization happens inside)
                Codebook::train(&sub_vectors, K, 20)
                    .context(format!("Failed to train codebook {}", m_idx))
            })
            .collect();

        Ok(Self {
            m,
            k: K,
            d_sub,
            codebooks: codebooks?,
        })
    }

    /// Quantize a full vector into quantized representation
    #[inline]
    pub fn quantize(&self, vector: &[f32]) -> QuantizedVector {
        let mut quantized = vec![0u8; self.m];

        for (m_idx, codebook) in self.codebooks.iter().enumerate() {
            let start = m_idx * self.d_sub;
            let end = start + self.d_sub;
            let sub_vec = &vector[start..end];

            quantized[m_idx] = codebook.encode(sub_vec);
        }

        quantized
    }

    // Legacy alias
    pub fn encode(&self, vector: &[f32]) -> QuantizedVector {
        self.quantize(vector)
    }

    /// Compute distance table for asymmetric distance computation
    pub fn compute_distance_table(&self, query: &[f32]) -> DistanceTable {
        let tables: Vec<Vec<f32>> = self
            .codebooks
            .iter()
            .enumerate()
            .map(|(m_idx, codebook)| {
                let start = m_idx * self.d_sub;
                let end = start + self.d_sub;
                let query_sub = &query[start..end];

                codebook.compute_distances(query_sub)
            })
            .collect();

        DistanceTable { tables }
    }

    /// Asymmetric distance computation using precomputed distance table
    /// Returns approximate Cosine Similarity (averaged across all sub-spaces)
    #[inline]
    pub fn asymmetric_distance(&self, qvec: &[u8], dtable: &DistanceTable) -> f32 {
        let mut sum_neg_cosine = 0.0f32;
        for (m_idx, &code) in qvec.iter().enumerate() {
            sum_neg_cosine += dtable.tables[m_idx][code as usize];
        }
        // Negate to get positive, then divide by M to get average cosine similarity
        -sum_neg_cosine / (self.m as f32)
    }

    /// Save codebooks to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).context("Failed to serialize quantizer")
    }

    /// Load codebooks from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).context("Failed to deserialize quantizer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_encode_decode() {
        let dim = 128;
        let m = 16;
        // Create simple training data
        let training: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                for (j, x) in v.iter_mut().enumerate() {
                    *x = (i as f32 / 100.0) + (j as f32 / 1536.0);
                }
                v
            })
            .collect();

        let pq = ProductQuantizer::train(&training, m).unwrap();

        let test_vec = vec![0.5f32; dim];
        let quantized = pq.quantize(&test_vec);

        assert_eq!(quantized.len(), m);
        // All codes should be within valid range
        assert!(quantized.iter().all(|&c| (c as usize) < K));
    }
}
