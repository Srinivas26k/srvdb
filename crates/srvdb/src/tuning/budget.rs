//! Resident-memory estimates and the memory-budget ladder (ADR 0018)
//!
//! Pure arithmetic over the storage layouts; no I/O. `D` is the dimension
//! and `n` the number of stored records, tombstones included.

use crate::quant::rabitq::fastscan::BLOCK;

const BLOCK_VECTORS: u64 = BLOCK as u64;

/// Share of the budget the estimate may fill: the rest covers growth slack
/// of the RaBitQ scan copy (at most 25 %, `1 / 1.25 = 0.8`) and what the
/// model does not count
pub(crate) const HEADROOM_PERCENT: u64 = 80;
/// Largest `memory_mb` accepted (1 TiB)
pub(crate) const MAX_MEMORY_MB: u64 = 1 << 20;
/// Estimated bytes per link (two `BTreeSet<(u16, u64)>` entries)
pub(crate) const LINK_BYTES: u64 = 64;
/// Estimated bytes per keyword posting and per indexed document
pub(crate) const KEYWORD_BYTES: u64 = 64;

/// A rung of the budget ladder, highest fidelity first
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tier {
    Flat,
    Rabitq(u8),
}

/// Flat, then RaBitQ 4, 2 and 1 bits: every rung keeps `vectors.bin`
pub(crate) const LADDER: [Tier; 4] = [
    Tier::Flat,
    Tier::Rabitq(4),
    Tier::Rabitq(2),
    Tier::Rabitq(1),
];

fn d(dim: usize) -> u64 {
    dim as u64
}

/// Bytes of one RaBitQ code (without the factor)
pub(crate) fn rabitq_code_bytes(dim: usize, bits: u8) -> u64 {
    d(dim).div_ceil(8 / u64::from(bits))
}

/// Resident bytes of the vectors (or codes) of `n` records in `tier`
pub(crate) fn tier_vector_bytes(tier: Tier, dim: usize, n: u64) -> u64 {
    match tier {
        Tier::Flat => n.saturating_mul(4 * d(dim)),
        Tier::Rabitq(bits) => {
            let padded = n.div_ceil(BLOCK_VECTORS) * BLOCK_VECTORS;
            padded
                .saturating_mul(rabitq_code_bytes(dim, bits))
                // factor and |g| width, 4 bytes each (ADR 0012, 0017)
                .saturating_add(n.saturating_mul(8))
        }
    }
}

/// Bytes `tier` holds whatever the count: the RaBitQ rotation
pub(crate) fn tier_fixed_bytes(tier: Tier, dim: usize) -> u64 {
    match tier {
        Tier::Flat => 0,
        Tier::Rabitq(_) => 4 * d(dim) * d(dim),
    }
}

/// Everything `tier` needs at `n` records
pub(crate) fn tier_bytes(tier: Tier, dim: usize, n: u64) -> u64 {
    tier_fixed_bytes(tier, dim).saturating_add(tier_vector_bytes(tier, dim, n))
}

/// SQ8: one byte per value, min and max per dimension
pub(crate) fn sq8_bytes(dim: usize, n: u64) -> (u64, u64) {
    (n.saturating_mul(d(dim)), 8 * d(dim))
}

/// PQ: `m` code bytes, 256 centroids of `D / m` floats per sub-space
pub(crate) fn pq_bytes(dim: usize, m: usize, n: u64) -> (u64, u64) {
    (n.saturating_mul(m as u64), 1024 * d(dim))
}

/// HNSW graph on top of the vectors: a level-0 list of `2m` ids, layer
/// `Vec`s and the hash-map slot
pub(crate) fn hnsw_bytes(m: usize, n: u64) -> u64 {
    n.saturating_mul(16 * m as u64 + 96)
}

/// The byte limit for a budget of `memory_mb` MiB
pub(crate) fn limit_bytes(memory_mb: u64) -> u64 {
    memory_mb.saturating_mul(1 << 20) / 100 * HEADROOM_PERCENT
}

/// The highest-fidelity rung at or below `from` that holds `n` records
/// next to `other` bytes within `limit`, or `None` if not even 1 bit fits.
/// The ladder is only walked down (ADR 0018).
pub(crate) fn select(from: Tier, dim: usize, n: u64, other: u64, limit: u64) -> Option<Tier> {
    let start = LADDER.iter().position(|&t| t == from).unwrap_or(0);
    LADDER[start..]
        .iter()
        .copied()
        .find(|&t| tier_bytes(t, dim, n).saturating_add(other) <= limit)
}

/// How many records `tier` holds within `limit` next to `other` bytes
pub(crate) fn capacity(tier: Tier, dim: usize, other: u64, limit: u64) -> u64 {
    let fits = |n: u64| tier_bytes(tier, dim, n).saturating_add(other) <= limit;
    if !fits(0) {
        return 0;
    }
    // Largest n that fits: grow an upper bound, then bisect
    let (mut lo, mut hi) = (0u64, 1u64);
    while fits(hi) {
        lo = hi;
        if hi > u64::MAX / 4 {
            return hi;
        }
        hi *= 2;
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
