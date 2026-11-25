//! Fast-scan of RaBitQ codes (ADR 0012)
//!
//! Codes are kept in memory in blocks of [`BLOCK`] = 32 vectors, byte `j`
//! of every vector side by side: `block[j * 32 + v]` is code byte `j` of
//! vector `v`. Each nibble of a code byte covers `4 / bits` dimensions,
//! so one 16-entry table per nibble scores every bit width. The tables
//! are quantized to u8 with one shared step, the kernels look them up in
//! registers (`pshufb` / `tbl`) and add in u16, widening to u32 every
//! [`CHUNK`] code bytes so nothing overflows. All kernels return the
//! same integer sums; the f32 estimate is `factor * (step * sum + bias)`.
//!
//! This copy is derived from `rabitq_codes.bin` on open and kept up to
//! date on append and truncate; the file format is unchanged.

mod avx2;
mod lut;
mod neon;
mod scalar;

pub(crate) use lut::QueryLut;

use crate::error::{Error, Result};
use std::collections::TryReserveError;

/// Vectors per interleaved block
pub(crate) const BLOCK: usize = 32;

/// Code bytes added in u16 before widening: each byte adds at most
/// 2 * 255, so 64 bytes stay below 65536
pub(crate) const CHUNK: usize = 64;

/// A scan kernel; [`Kernel::detect`] picks the best one the CPU supports.
///
/// The variant is private, so safe code can only get a SIMD kernel from
/// `detect()`, after the CPU feature check that makes it sound to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Kernel(Inner);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Inner {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Kernel {
    /// The portable kernel, always available
    pub(crate) const SCALAR: Kernel = Kernel(Inner::Scalar);

    /// Best kernel for this CPU (feature detection is cached by std)
    pub(crate) fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return Kernel(Inner::Avx2);
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Kernel(Inner::Neon);
        }
        Self::SCALAR
    }

    /// Every kernel this CPU can run (for tests)
    #[cfg(test)]
    pub(crate) fn available() -> Vec<Self> {
        let mut all = vec![Self::SCALAR];
        if Self::detect() != Self::SCALAR {
            all.push(Self::detect());
        }
        all
    }

    /// Integer sums of one block; `luts` holds 32 bytes per code byte
    /// (low-nibble table, then high-nibble table)
    pub(crate) fn block_sums(self, block: &[u8], luts: &[u8], code_bytes: usize) -> [u32; BLOCK] {
        assert!(block.len() >= code_bytes * BLOCK && luts.len() >= code_bytes * 32);
        match self.0 {
            Inner::Scalar => scalar::block_sums(block, luts, code_bytes),
            // SAFETY: Inner::Avx2 is private to this module and only built
            // by detect() after is_x86_feature_detected!("avx2"); the
            // slice lengths the kernel reads were asserted above
            #[cfg(target_arch = "x86_64")]
            Inner::Avx2 => unsafe { avx2::block_sums(block, luts, code_bytes) },
            // SAFETY: Inner::Neon is private and only built by detect()
            // after NEON detection; slice lengths were asserted above
            #[cfg(target_arch = "aarch64")]
            Inner::Neon => unsafe { neon::block_sums(block, luts, code_bytes) },
        }
    }
}

/// Make room for `extra` more items, growing by at least 25 % so appends
/// stay amortised O(1) while capacity never exceeds 1.25x the length
/// (`Vec`'s own doubling could leave twice the memory budget, ADR 0018)
fn grow<T>(v: &mut Vec<T>, extra: usize) -> std::result::Result<(), TryReserveError> {
    let needed = v.len().saturating_add(extra);
    if needed <= v.capacity() {
        return Ok(());
    }
    let target = needed.max(v.len() + v.len() / 4);
    v.try_reserve_exact(target - v.len())
}

/// Interleaved codes and correction factors of every stored vector
#[derive(Debug, Clone)]
pub(crate) struct FastScanCodes {
    code_bytes: usize,
    blocks: Vec<u8>,
    factors: Vec<f32>,
    /// `(dim, bits)` when the per-vector error widths are kept
    grid: Option<(usize, u8)>,
    /// Unit error width of each vector (ADR 0017), empty without `grid`
    widths: Vec<f32>,
}

impl FastScanCodes {
    pub(crate) fn new(code_bytes: usize) -> Self {
        Self {
            code_bytes,
            blocks: Vec::new(),
            factors: Vec::new(),
            grid: None,
            widths: Vec::new(),
        }
    }

    /// Codes of `dim` dimensions at `bits` that also keep the per-vector
    /// error widths certified search needs (ADR 0017)
    pub(crate) fn with_widths(dim: usize, bits: u8) -> Self {
        let mut codes = Self::new(super::record_size(dim, bits) - super::FACTOR_BYTES);
        codes.grid = Some((dim, bits));
        codes
    }

    /// Correction factor of vector `i`
    pub(crate) fn factor(&self, i: usize) -> f32 {
        self.factors[i]
    }

    /// Unit error width of vector `i`, `None` if widths are not kept
    pub(crate) fn width(&self, i: usize) -> Option<f32> {
        self.widths.get(i).copied()
    }

    pub(crate) fn len(&self) -> usize {
        self.factors.len()
    }

    /// Bytes of one block
    pub(crate) fn block_bytes(&self) -> usize {
        self.code_bytes * BLOCK
    }

    /// Append records (`FACTOR_BYTES` + codes each, `record_size` long).
    ///
    /// # Errors
    /// `Io` with kind `OutOfMemory` if memory for the copy cannot be
    /// reserved (the codes are held in memory a second time, next to the
    /// mapped file). Not `InvalidState`: that would make the engine
    /// delete and rebuild the codes, which cannot help.
    pub(crate) fn extend(&mut self, records: &[u8], record_size: usize) -> Result<()> {
        let block_bytes = self.block_bytes();
        self.reserve(records.len() / record_size.max(1))?;
        for record in records.chunks_exact(record_size) {
            let (factor, codes) = record.split_at(super::FACTOR_BYTES);
            let slot = self.len() % BLOCK;
            if slot == 0 {
                self.blocks.resize(self.blocks.len() + block_bytes, 0);
            }
            let start = self.blocks.len() - block_bytes;
            let block = &mut self.blocks[start..];
            for (j, &b) in codes.iter().enumerate().take(self.code_bytes) {
                block[j * BLOCK + slot] = b;
            }
            let factor = f32::from_le_bytes([factor[0], factor[1], factor[2], factor[3]]);
            self.factors.push(factor);
            if let Some((dim, bits)) = self.grid {
                self.widths
                    .push(super::certify::unit_width(factor, codes, dim, bits));
            }
        }
        Ok(())
    }

    /// Reserve room for `new` more vectors without aborting on failure
    fn reserve(&mut self, new: usize) -> Result<()> {
        let total = self
            .len()
            .checked_add(new)
            .and_then(|n| n.div_ceil(BLOCK).checked_mul(self.block_bytes()));
        let reserved = total.is_some_and(|bytes| {
            let extra = bytes.saturating_sub(self.blocks.len());
            grow(&mut self.blocks, extra)
                .and_then(|()| grow(&mut self.factors, new))
                .and_then(|()| match self.grid {
                    Some(_) => grow(&mut self.widths, new),
                    None => Ok(()),
                })
                .is_ok()
        });
        if !reserved {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!(
                    "not enough memory for the in-memory RaBitQ scan copy of {new} \
                     codes of {} bytes; free memory or use another index mode",
                    self.code_bytes
                ),
            )));
        }
        Ok(())
    }

    /// Keep the first `n` vectors
    pub(crate) fn truncate(&mut self, n: usize) {
        if n >= self.len() {
            return;
        }
        self.factors.truncate(n);
        self.widths.truncate(n);
        self.blocks.truncate(n.div_ceil(BLOCK) * self.block_bytes());
        let slot = n % BLOCK;
        if slot != 0 {
            let start = self.blocks.len() - self.block_bytes();
            for j in 0..self.code_bytes {
                self.blocks[start + j * BLOCK + slot..start + (j + 1) * BLOCK].fill(0);
            }
        }
    }

    /// Estimated similarities of block `index` (only `len` are real)
    pub(crate) fn block_scores(
        &self,
        index: usize,
        lut: &QueryLut,
        kernel: Kernel,
        out: &mut [f32; BLOCK],
    ) -> usize {
        let bytes = self.block_bytes();
        let block = &self.blocks[index * bytes..(index + 1) * bytes];
        let sums = kernel.block_sums(block, lut.bytes(), self.code_bytes);
        let first = index * BLOCK;
        let real = (self.len() - first).min(BLOCK);
        for v in 0..real {
            out[v] = self.factors[first + v] * lut.estimate(sums[v]);
        }
        real
    }

    /// Estimated similarities of every vector, in id order (for tests)
    #[cfg(test)]
    pub(crate) fn scores(&self, lut: &QueryLut, kernel: Kernel) -> Vec<f32> {
        let mut all = Vec::with_capacity(self.len());
        let mut out = [0.0; BLOCK];
        for b in 0..self.len().div_ceil(BLOCK) {
            let real = self.block_scores(b, lut, kernel, &mut out);
            all.extend_from_slice(&out[..real]);
        }
        all
    }
}

#[cfg(test)]
mod tests;
