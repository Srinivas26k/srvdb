//! Public configuration of the RaBitQ mode

use crate::error::{Error, Result};

/// Bit widths a RaBitQ code may use per dimension
pub const SUPPORTED_BITS: [u8; 3] = [1, 2, 4];

/// Default rerank factor for a bit width: fewer bits estimate less
/// precisely, so more candidates are rescored (1 bit: 40, 2 bits: 20,
/// 4 bits: 10). Chosen from the recall tests in `tests/test_rabitq.rs`.
pub fn default_rerank(bits: u8) -> usize {
    match bits {
        1 => 40,
        2 => 20,
        _ => 10,
    }
}

/// Default rotation seed; any fixed value works, this one is documented
pub const DEFAULT_SEED: u64 = 0x5352_5644_4252_5131; // "SRVDBRQ1"

/// Settings for [`crate::SrvDB::new_rabitq`] (ADR 0008).
///
/// Build with [`RabitqConfig::new`] and the `with_*` methods; the struct
/// is `#[non_exhaustive]` so fields may be added later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RabitqConfig {
    /// Bits per dimension: 1, 2 or 4
    pub bits: u8,
    /// Rerank factor `r`: the code scan keeps `k * r` candidates and
    /// rescores them exactly against the stored f32 vectors. `0` disables
    /// reranking and returns the estimated cosine similarities.
    pub rerank: usize,
    /// Seed of the random rotation applied before quantizing
    pub seed: u64,
}

impl RabitqConfig {
    /// Configuration with `bits` per dimension, the default rerank factor
    /// ([`default_rerank`]) and the default seed.
    ///
    /// # Errors
    /// `InvalidInput` if `bits` is not 1, 2 or 4.
    pub fn new(bits: u8) -> Result<Self> {
        let config = Self {
            bits,
            rerank: default_rerank(bits),
            seed: DEFAULT_SEED,
        };
        config.validate()?;
        Ok(config)
    }

    /// Set the rerank factor (`0` = no rerank)
    pub fn with_rerank(mut self, rerank: usize) -> Self {
        self.rerank = rerank;
        self
    }

    /// Set the rotation seed. Changing it on an existing database makes
    /// SrvDB rebuild the codes from the stored vectors on open.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Check the settings.
    ///
    /// # Errors
    /// `InvalidInput` if `bits` is not 1, 2 or 4, or `rerank` is so large
    /// that `k * rerank` could overflow.
    pub fn validate(&self) -> Result<()> {
        if !SUPPORTED_BITS.contains(&self.bits) {
            return Err(Error::invalid_input(format!(
                "RaBitQ supports 1, 2 or 4 bits per dimension, got {}",
                self.bits
            )));
        }
        if self.rerank > 1_000_000 {
            return Err(Error::invalid_input(format!(
                "rerank factor {} is too large (at most 1000000)",
                self.rerank
            )));
        }
        Ok(())
    }
}
