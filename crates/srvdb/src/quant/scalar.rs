//! Scalar quantization (SQ8): per-dimension min/max scaled to u8

use serde::{Deserialize, Serialize};

/// Scalar Quantization (SQ8) - Simple 4x compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    pub min_vals: Vec<f32>,
    pub max_vals: Vec<f32>,
    pub dimension: usize,
}

impl ScalarQuantizer {
    /// Train scalar quantizer by finding min/max per dimension
    pub fn train(training_data: &[Vec<f32>]) -> anyhow::Result<Self> {
        if training_data.is_empty() {
            anyhow::bail!("Training data cannot be empty");
        }

        let dimension = training_data[0].len();
        let mut min_vals = vec![f32::MAX; dimension];
        let mut max_vals = vec![f32::MIN; dimension];

        for vec in training_data {
            if vec.len() != dimension {
                anyhow::bail!("Inconsistent dimensions in training data");
            }

            for (i, &val) in vec.iter().enumerate() {
                min_vals[i] = min_vals[i].min(val);
                max_vals[i] = max_vals[i].max(val);
            }
        }

        Ok(Self {
            min_vals,
            max_vals,
            dimension,
        })
    }

    /// Encode float32 vector to uint8 (4x compression)
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let min = self.min_vals[i];
                let max = self.max_vals[i];
                let range = max - min;

                if range < 1e-10 {
                    return 0u8;
                }

                let normalized = ((val - min) / range * 255.0).clamp(0.0, 255.0);
                normalized as u8
            })
            .collect()
    }

    /// Decode uint8 vector back to approximate float32
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .enumerate()
            .map(|(i, &code)| {
                let min = self.min_vals[i];
                let max = self.max_vals[i];
                let range = max - min;

                min + (code as f32 / 255.0) * range
            })
            .collect()
    }

    /// Asymmetric distance: compare quantized vector to full-precision query
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let decoded = self.decode(codes);

        // Compute cosine similarity
        let dot: f32 = query.iter().zip(decoded.iter()).map(|(a, b)| a * b).sum();

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let decoded_norm: f32 = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();

        if query_norm < 1e-10 || decoded_norm < 1e-10 {
            return 0.0;
        }

        dot / (query_norm * decoded_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantizer() {
        let training: Vec<Vec<f32>> = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.1, 0.6, 0.9],
            vec![0.2, 0.4, 1.0],
        ];

        let sq = ScalarQuantizer::train(&training).unwrap();

        let test_vec = vec![0.15, 0.55, 0.95];
        let encoded = sq.encode(&test_vec);
        assert_eq!(encoded.len(), 3);

        let decoded = sq.decode(&encoded);

        // Check reconstruction error is small
        for (orig, dec) in test_vec.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.1);
        }
    }
}
