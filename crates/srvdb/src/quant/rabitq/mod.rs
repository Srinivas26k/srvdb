//! RaBitQ-style low-bit quantization, training-free (ADR 0008)
//!
//! Encoding a vector `x` with `b` bits per dimension:
//! 1. `y = R x / |x|`, with `R` a seeded random orthogonal rotation, so
//!    `y` is a unit vector whose coordinates all look alike.
//! 2. Codes `c_i in [0, 2^b)` place `t * y_i` on the grid
//!    `g_i = c_i + 1/2 - 2^(b-1)`; the scale `t` is chosen per vector to
//!    maximise `<g, y> / |g|` (for `b = 1` the grid is just the signs).
//! 3. One correction factor `f = 1 / <g, y>` is stored with the codes.
//!
//! For a unit query `q' = R q / |q|` the cosine similarity is estimated as
//! `f * <g, q'>` = `<g, q'> / <g, y>`, the RaBitQ estimator
//! `<o_bar, q> / <o_bar, o>`, which is unbiased under the random rotation.
//!
//! Record layout (little-endian): `f` as f32, then the codes packed
//! `8 / b` dimensions per byte, lowest bits first, dimensions in order.
//! The scan scores a record with one 16-entry table per nibble (a nibble
//! is `4 / b` dimensions), so every bit width shares the fast-scan kernel
//! in [`fastscan`] (ADR 0012).

pub(crate) mod certify;
mod config;
pub(crate) mod fastscan;
mod outcome;
mod rotation;
pub(crate) mod scan;

pub use certify::CERTIFIED_CONFIDENCE;
pub use config::{default_rerank, RabitqConfig, DEFAULT_SEED, MAX_DIMENSION, SUPPORTED_BITS};
pub use outcome::{RabitqSearch, SearchOutcome, SearchStats, DEFAULT_MAX_RERANK};
pub(crate) use rotation::Rotation;

/// Bytes before the packed codes in a record (the f32 factor)
pub(crate) const FACTOR_BYTES: usize = 4;

/// Candidate grid scales, as multiples of `2^(b-1) / max|y_i|`
const SCALES: usize = 32;

/// Encoder/scorer for one dimension, bit width and rotation
#[derive(Debug, Clone)]
pub(crate) struct RabitqQuantizer {
    dim: usize,
    bits: u8,
    rotation: Rotation,
}

/// A query prepared for scanning: one 16-entry f32 table per nibble
pub(crate) struct RabitqQuery {
    tables: Vec<[f32; 16]>,
}

/// Bytes per stored vector for `dim` dimensions at `bits` (one of
/// [`SUPPORTED_BITS`]); the same as [`RabitqQuantizer::record_size`]
/// without building the rotation
pub(crate) fn record_size(dim: usize, bits: u8) -> usize {
    FACTOR_BYTES + dim.div_ceil(8 / bits as usize)
}

impl RabitqQuantizer {
    /// `bits` must be one of [`SUPPORTED_BITS`] (checked by the config)
    pub(crate) fn new(dim: usize, bits: u8, seed: u64) -> Self {
        // Other widths would mis-pack codes; RabitqConfig::validate and
        // the header mode byte keep them out
        debug_assert!(SUPPORTED_BITS.contains(&bits), "unsupported bits {bits}");
        Self {
            dim,
            bits,
            rotation: Rotation::new(dim, seed),
        }
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn bits(&self) -> u8 {
        self.bits
    }

    pub(crate) fn rotation_fingerprint(&self) -> u32 {
        self.rotation.fingerprint()
    }

    fn per_byte(&self) -> usize {
        8 / self.bits as usize
    }

    pub(crate) fn code_bytes(&self) -> usize {
        self.dim.div_ceil(self.per_byte())
    }

    /// Bytes per stored vector
    pub(crate) fn record_size(&self) -> usize {
        record_size(self.dim, self.bits)
    }

    /// Unit vector rotated; zero (or non-finite norm) maps to zeros
    fn rotate_unit(&self, x: &[f32]) -> Vec<f32> {
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return vec![0.0; self.dim];
        }
        let unit: Vec<f32> = x.iter().map(|v| v / norm).collect();
        self.rotation.apply(&unit)
    }

    /// Grid codes for `y` at scale `t`
    fn grid(&self, y: &[f32], t: f32) -> Vec<u8> {
        let levels = 1u32 << self.bits;
        let half = (levels / 2) as f32;
        let top = (levels - 1) as f32;
        y.iter()
            .map(|v| (v * t + half).floor().clamp(0.0, top) as u8)
            .collect()
    }

    fn offset(&self) -> f32 {
        0.5 - (1u32 << (self.bits - 1)) as f32
    }

    /// Append the record for `x` to `out`
    pub(crate) fn encode_into(&self, x: &[f32], out: &mut Vec<u8>) {
        let y = self.rotate_unit(x);
        let max = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let base = (1u32 << (self.bits - 1)) as f32 / max.max(f32::MIN_POSITIVE);
        let offset = self.offset();
        let mut best: Option<(f32, f32, Vec<u8>)> = None;
        let steps = if self.bits == 1 { 1 } else { SCALES };
        for s in 1..=steps {
            // Scales from 0.25x to 2x of the no-clipping scale
            let t = base * (0.25 + 1.75 * s as f32 / steps as f32);
            let codes = self.grid(&y, t);
            let (mut gy, mut gg) = (0.0f32, 0.0f32);
            for (c, v) in codes.iter().zip(&y) {
                let g = *c as f32 + offset;
                gy += g * v;
                gg += g * g;
            }
            let quality = gy / gg.sqrt();
            if best.as_ref().is_none_or(|b| quality > b.0) {
                best = Some((quality, gy, codes));
            }
        }
        let (_, gy, codes) = best.unwrap_or((0.0, 0.0, vec![0; self.dim]));
        let factor = if gy > 0.0 { 1.0 / gy } else { 0.0 };
        out.extend_from_slice(&factor.to_le_bytes());
        let per = self.per_byte();
        for chunk in codes.chunks(per) {
            let byte = chunk
                .iter()
                .enumerate()
                .fold(0u8, |b, (k, c)| b | (c << (k * self.bits as usize)));
            out.push(byte);
        }
    }

    /// Build the nibble tables for `query`: for code byte `j`, the
    /// partial dot product of every low-nibble value, then of every
    /// high-nibble value (a nibble covers `4 / bits` dimensions)
    pub(crate) fn prepare(&self, query: &[f32]) -> RabitqQuery {
        let q = self.rotate_unit(query);
        let bits = self.bits as usize;
        let per_nibble = 4 / bits;
        let mask = (1usize << bits) - 1;
        let offset = self.offset();
        let mut tables = vec![[0.0f32; 16]; 2 * self.code_bytes()];
        for (m, table) in tables.iter_mut().enumerate() {
            let first = (m * per_nibble).min(self.dim);
            let dims = &q[first..((m + 1) * per_nibble).min(self.dim)];
            for (nibble, slot) in table.iter_mut().enumerate() {
                *slot = dims
                    .iter()
                    .enumerate()
                    .map(|(k, qv)| (((nibble >> (k * bits)) & mask) as f32 + offset) * qv)
                    .sum();
            }
        }
        RabitqQuery { tables }
    }
}

impl RabitqQuery {
    /// The f32 nibble tables, two per code byte
    pub(crate) fn tables(&self) -> &[[f32; 16]] {
        &self.tables
    }

    /// Estimated cosine similarity of the query and one record in f32
    /// (the reference the u8 fast-scan is tested against)
    #[cfg(test)]
    pub(crate) fn score(&self, record: &[u8]) -> f32 {
        let (factor, codes) = record.split_at(FACTOR_BYTES);
        let factor = f32::from_le_bytes([factor[0], factor[1], factor[2], factor[3]]);
        let dot: f32 = codes
            .iter()
            .enumerate()
            .map(|(j, &b)| {
                self.tables[2 * j][(b & 15) as usize] + self.tables[2 * j + 1][(b >> 4) as usize]
            })
            .sum();
        factor * dot
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
