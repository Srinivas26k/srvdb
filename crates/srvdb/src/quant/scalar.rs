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
        self.scorer(query).score(codes)
    }

    /// Prepare to score one query against many code vectors.
    ///
    /// Computes the query norm and the per-dimension decode step once, so a
    /// scan does not redo that work for every code.
    pub(crate) fn scorer<'a>(&'a self, query: &'a [f32]) -> CodeScorer<'a> {
        const INV_255: f32 = 1.0 / 255.0;
        let step = self
            .min_vals
            .iter()
            .zip(&self.max_vals)
            .map(|(&lo, &hi)| (hi - lo) * INV_255)
            .collect();
        CodeScorer {
            query,
            query_norm: query.iter().map(|x| x * x).sum::<f32>().sqrt(),
            min_vals: &self.min_vals,
            step,
        }
    }
}

/// Cosine scorer for one full-precision query against SQ8 codes.
pub(crate) struct CodeScorer<'a> {
    query: &'a [f32],
    query_norm: f32,
    min_vals: &'a [f32],
    /// Decoded value per code unit: `(max - min) / 255` for each dimension
    step: Vec<f32>,
}

impl CodeScorer<'_> {
    /// Cosine similarity between the query and one code vector.
    ///
    /// One pass, no allocation: each code is decoded on the fly while the dot
    /// product and the decoded norm accumulate in independent lanes, which
    /// lets the compiler use SIMD. Results agree with decoding first and then
    /// calling cosine to within floating-point rounding (about 1e-6).
    #[inline]
    pub(crate) fn score(&self, codes: &[u8]) -> f32 {
        const LANES: usize = 16;
        let mut dot = [0.0f32; LANES];
        let mut sq = [0.0f32; LANES];

        let (q_chunks, q_rem) = self.query.as_chunks::<LANES>();
        let (c_chunks, c_rem) = codes.as_chunks::<LANES>();
        let (lo_chunks, lo_rem) = self.min_vals.as_chunks::<LANES>();
        let (s_chunks, s_rem) = self.step.as_chunks::<LANES>();

        for (((q, c), lo), s) in q_chunks.iter().zip(c_chunks).zip(lo_chunks).zip(s_chunks) {
            for l in 0..LANES {
                let decoded = lo[l] + c[l] as f32 * s[l];
                dot[l] += q[l] * decoded;
                sq[l] += decoded * decoded;
            }
        }

        let mut dot_sum: f32 = dot.iter().sum();
        let mut sq_sum: f32 = sq.iter().sum();
        for (((&q, &c), &lo), &s) in q_rem.iter().zip(c_rem).zip(lo_rem).zip(s_rem) {
            let decoded = lo + c as f32 * s;
            dot_sum += q * decoded;
            sq_sum += decoded * decoded;
        }

        let decoded_norm = sq_sum.sqrt();
        if self.query_norm < 1e-10 || decoded_norm < 1e-10 {
            return 0.0;
        }

        dot_sum / (self.query_norm * decoded_norm)
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

    /// The original implementation, kept as the oracle for the single-pass version
    fn reference_distance(sq: &ScalarQuantizer, query: &[f32], codes: &[u8]) -> f32 {
        let decoded = sq.decode(codes);
        let dot: f32 = query.iter().zip(decoded.iter()).map(|(a, b)| a * b).sum();
        let qn: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dn: f32 = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
        if qn < 1e-10 || dn < 1e-10 {
            return 0.0;
        }
        dot / (qn * dn)
    }

    /// Deterministic values in [-1, 1) from a linear congruential generator
    fn lcg_vectors(n: usize, dim: usize, mut seed: u64) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((seed >> 33) as f32 / (1u64 << 30) as f32) - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn single_pass_distance_matches_the_original() {
        let data = lcg_vectors(200, 384, 7);
        let sq = ScalarQuantizer::train(&data).unwrap();
        let query = &data[0];
        for v in &data {
            let codes = sq.encode(v);
            let got = sq.asymmetric_distance(query, &codes);
            let want = reference_distance(&sq, query, &codes);
            assert!((got - want).abs() <= 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn zero_query_and_constant_dimension_score_zero() {
        let sq = ScalarQuantizer::train(&[vec![0.5; 4], vec![0.5; 4]]).unwrap();
        let codes = sq.encode(&[0.5; 4]);
        assert_eq!(sq.asymmetric_distance(&[0.0; 4], &codes), 0.0);
    }

    /// Run with: cargo test --release -p srvdb --lib scan_speed -- --ignored --nocapture
    #[test]
    #[ignore = "timing measurement, not a correctness test"]
    fn scan_speed_single_pass_vs_original() {
        use std::time::Instant;
        let data = lcg_vectors(4500, 384, 11);
        let sq = ScalarQuantizer::train(&data).unwrap();
        let codes: Vec<Vec<u8>> = data.iter().map(|v| sq.encode(v)).collect();
        let query = &data[1];
        let rounds = 20;

        let t = Instant::now();
        let mut a = 0.0f32;
        for _ in 0..rounds {
            for c in &codes {
                a += reference_distance(&sq, query, c);
            }
        }
        let original = t.elapsed();

        let t = Instant::now();
        let mut b = 0.0f32;
        let scorer = sq.scorer(query);
        for _ in 0..rounds {
            for c in &codes {
                b += scorer.score(c);
            }
        }
        let fused = t.elapsed();
        assert!((a - b).abs() / a.abs().max(1.0) < 1e-4);
        println!(
            "per full scan of 4500 vectors: original {:.2} ms, single pass {:.2} ms ({:.1}x)",
            original.as_secs_f64() * 1e3 / rounds as f64,
            fused.as_secs_f64() * 1e3 / rounds as f64,
            original.as_secs_f64() / fused.as_secs_f64()
        );
    }
}
