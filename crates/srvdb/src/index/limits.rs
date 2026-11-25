//! Bounds on index parameters, shared by the public API and the loaders
//! of on-disk settings (ADR 0011, issue #48)
//!
//! A parameter outside these bounds is refused before anything is
//! allocated: through the API as `Error::InvalidInput`, from a file as
//! `Error::Corrupt`. They are far above useful values; their job is to
//! keep a typo or a crafted file from allocating unbounded memory.

use crate::error::{Error, Result};

/// Largest vector dimension (the vector header and `DatabaseConfig` limit)
pub const MAX_DIMENSION: usize = 4096;
/// Smallest dimension `DatabaseConfig::new` and the quantized constructors
/// accept for a new database
pub const MIN_DIMENSION: usize = 128;
/// Largest HNSW `m` (links per node; layer 0 keeps `2 * m`)
pub const MAX_HNSW_M: usize = 256;
/// Smallest HNSW `m`: the level multiplier is `1 / ln(m)`
pub const MIN_HNSW_M: usize = 2;
/// Largest HNSW `ef_construction` and `ef_search`
pub const MAX_HNSW_EF: usize = 10_000;
/// Largest IVF `nlist` (partitions) and `nprobe`
pub const MAX_IVF_NLIST: usize = 65_536;
/// Largest number of k-means iterations when training IVF
pub const MAX_IVF_ITERATIONS: usize = 1_000;

/// `value` in `min..=max`, or `InvalidInput` naming the parameter
pub(crate) fn check(name: &str, value: usize, min: usize, max: usize) -> Result<()> {
    if value < min || value > max {
        return Err(Error::invalid_input(format!(
            "{name} must be between {min} and {max}, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_are_inclusive() {
        assert!(check("m", 2, 2, 256).is_ok());
        assert!(check("m", 256, 2, 256).is_ok());
        assert!(matches!(check("m", 1, 2, 256), Err(Error::InvalidInput(_))));
        assert!(matches!(
            check("m", 257, 2, 256),
            Err(Error::InvalidInput(_))
        ));
    }
}
