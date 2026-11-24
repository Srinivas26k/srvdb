//! Product Quantization (PQ) for vector compression
//!
//! Implements FAISS-style Product Quantization:
//! - ~32x memory compression (float32 -> uint8)
//! - Dynamic dimension support
//! - Asymmetric Distance Computation (ADC) for fast search
//! - K-means clustering for codebook training

use super::codebook::Codebook;
use crate::error::Error;
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
        let Some(first) = training_data.first() else {
            return Err(Error::invalid_input("at least one training vector is required").into());
        };
        let dim = first.len();
        if m == 0 || !dim.is_multiple_of(m) {
            return Err(Error::invalid_input(format!(
                "PQ needs a sub-quantizer count that divides the dimension; \
                 {m} does not divide {dim}"
            ))
            .into());
        }
        let d_sub = dim / m;

        log::debug!(
            "Training Product Quantizer with M={}, K={}, D_sub={}",
            m,
            K,
            d_sub
        );

        // Train codebook for each sub-quantizer
        let codebooks: Result<Vec<Codebook>> = (0..m)
            .into_par_iter()
            .map(|m_idx| {
                log::trace!("training PQ sub-quantizer {}/{}", m_idx + 1, m);

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

    /// Approximate reconstruction of a quantized vector: the chosen
    /// centroid of every sub-space, concatenated. `None` if `codes` does not
    /// match this quantizer (wrong length or an unknown centroid index).
    pub fn decode(&self, codes: &[u8]) -> Option<Vec<f32>> {
        if codes.len() != self.m {
            return None;
        }
        let mut vector = Vec::with_capacity(self.m * self.d_sub);
        for (codebook, &code) in self.codebooks.iter().zip(codes) {
            vector.extend_from_slice(codebook.centroids.get(code as usize)?);
        }
        Some(vector)
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
        crate::storage::format::encode(crate::storage::format::PQ_CODEBOOKS, self)
    }

    /// Load codebooks from bytes (current format, or as written by v0.2)
    ///
    /// Fails if the codebooks do not form `m` sub-quantizers of `d_sub`
    /// floats each with at most 256 centroids: search would index past them.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let pq: Self = if crate::storage::format::is_enveloped(bytes) {
            crate::storage::format::decode(crate::storage::format::PQ_CODEBOOKS, bytes)?
        } else {
            crate::storage::legacy::product_quantizer(bytes)?
        };
        pq.check_shape()?;
        Ok(pq)
    }

    fn check_shape(&self) -> Result<()> {
        let dim = self.m.saturating_mul(self.d_sub);
        let consistent = self.m > 0
            && self.d_sub > 0
            && dim <= crate::index::limits::MAX_DIMENSION
            && self.codebooks.len() == self.m
            && self.codebooks.iter().all(|c| {
                (1..=K).contains(&c.centroids.len())
                    && c.centroids.iter().all(|v| v.len() == self.d_sub)
            });
        if !consistent {
            return Err(Error::Corrupt(format!(
                "PQ codebooks are inconsistent (m={}, d_sub={}, {} codebooks); \
                 restore codebooks.bin from a backup",
                self.m,
                self.d_sub,
                self.codebooks.len()
            ))
            .into());
        }
        Ok(())
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

    #[test]
    fn similarity_stays_within_cosine_range() {
        // Reseeded empty clusters keep raw, non-unit centroids: a long one
        // parallel to the query used to score above 1 (issue #51)
        let codebook = Codebook {
            centroids: vec![vec![3.0, 0.0], vec![-0.5, 0.0]],
        };
        let pq = ProductQuantizer {
            m: 2,
            k: 2,
            d_sub: 2,
            codebooks: vec![codebook.clone(), codebook],
        };
        let table = pq.compute_distance_table(&[1.0, 0.0, 1.0, 0.0]);
        let best = pq.asymmetric_distance(&[0, 0], &table);
        let worst = pq.asymmetric_distance(&[1, 1], &table);
        assert!((best - 1.0).abs() < 1e-6, "got {best}");
        assert!((worst + 1.0).abs() < 1e-6, "got {worst}");
    }

    #[test]
    fn train_rejects_m_that_does_not_divide_dimension() {
        let training = vec![vec![0.5f32; 12]; 4];
        for m in [0, 5] {
            let err = crate::Error::from(ProductQuantizer::train(&training, m).unwrap_err());
            assert!(
                matches!(err, crate::Error::InvalidInput(_)),
                "m={m}: {err:?}"
            );
        }
    }

    #[test]
    fn from_bytes_rejects_inconsistent_shape() {
        let codebook = Codebook {
            centroids: vec![vec![1.0, 0.0]; 2],
        };
        // Claims 4 sub-quantizers but stores 1: search would index past it
        let pq = ProductQuantizer {
            m: 4,
            k: 2,
            d_sub: 2,
            codebooks: vec![codebook],
        };
        let bytes = pq.to_bytes().unwrap();
        assert!(ProductQuantizer::from_bytes(&bytes).is_err());
    }
}
