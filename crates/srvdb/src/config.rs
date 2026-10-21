//! Database configuration types

use serde::{Deserialize, Serialize};

/// Database configuration with dimension support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub dimension: usize,
    pub quantization: QuantizationConfig,
    pub index_type: IndexType,
}

impl DatabaseConfig {
    pub fn new(dimension: usize) -> anyhow::Result<Self> {
        if !(128..=4096).contains(&dimension) {
            anyhow::bail!("Dimension must be between 128 and 4096, got {}", dimension);
        }

        Ok(Self {
            dimension,
            quantization: QuantizationConfig::default(),
            index_type: IndexType::Flat,
        })
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.quantization.enabled {
            // Ensure dimension is divisible by M for PQ
            if !self.dimension.is_multiple_of(self.quantization.m) {
                anyhow::bail!(
                    "Dimension {} must be divisible by M={} for Product Quantization",
                    self.dimension,
                    self.quantization.m
                );
            }
        }
        Ok(())
    }

    /// Auto-tune PQ parameters based on dimension
    pub fn auto_tune_pq(&mut self) {
        // Heuristic: M = dim / 8 (8 dimensions per subquantizer)
        let optimal_m = (self.dimension / 8).clamp(16, 256);
        self.quantization.m = optimal_m;
        self.quantization.d_sub = self.dimension / optimal_m;
    }
}

/// Index type selection
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IndexType {
    Flat,             // Brute force O(n)
    HNSW,             // Graph-based O(log n)
    ScalarQuantized,  // SQ8 compression (4x)
    ProductQuantized, // PQ compression (32x)
    HNSWQuantized,    // HNSW + PQ hybrid
    IVF,              // Inverted File with HNSW Refinement
}

/// Quantization configuration with multiple modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub mode: QuantizationMode,
    pub m: usize,     // Number of sub-quantizers (PQ only)
    pub k: usize,     // Centroids per sub-quantizer (PQ only)
    pub d_sub: usize, // Dimensions per sub-space (PQ only)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantizationMode {
    None,
    Scalar,  // SQ8: 4x compression, fast, 95%+ recall
    Product, // PQ: 32x compression, slower, 85-95% recall
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: QuantizationMode::None,
            m: 192, // Will be auto-tuned
            k: 256,
            d_sub: 8, // Will be auto-tuned
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_dimensions() {
        let config = DatabaseConfig::new(384).unwrap();
        assert_eq!(config.dimension, 384);

        // Test invalid dimensions
        assert!(DatabaseConfig::new(64).is_err());
        assert!(DatabaseConfig::new(5000).is_err());
    }

    #[test]
    fn test_auto_tune_pq() {
        let mut config = DatabaseConfig::new(768).unwrap();
        config.auto_tune_pq();

        // Should be divisible
        assert_eq!(768 % config.quantization.m, 0);
        assert_eq!(config.quantization.d_sub, 768 / config.quantization.m);
    }
}
