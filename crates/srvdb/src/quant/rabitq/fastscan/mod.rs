//! Fast-scan of RaBitQ codes (ADR 0011)
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

/// Vectors per interleaved block
pub(crate) const BLOCK: usize = 32;

/// Code bytes added in u16 before widening: each byte adds at most
/// 2 * 255, so 64 bytes stay below 65536
pub(crate) const CHUNK: usize = 64;

/// A scan kernel; [`Kernel::detect`] picks the best one the CPU supports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Kernel {
    /// Best kernel for this CPU (feature detection is cached by std)
    pub(crate) fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return Kernel::Avx2;
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Kernel::Neon;
        }
        Kernel::Scalar
    }

    /// Every kernel this CPU can run (for tests)
    #[cfg(test)]
    pub(crate) fn available() -> Vec<Self> {
        let mut all = vec![Kernel::Scalar];
        if Self::detect() != Kernel::Scalar {
            all.push(Self::detect());
        }
        all
    }

    /// Integer sums of one block; `luts` holds 32 bytes per code byte
    /// (low-nibble table, then high-nibble table)
    pub(crate) fn block_sums(self, block: &[u8], luts: &[u8], code_bytes: usize) -> [u32; BLOCK] {
        assert!(block.len() >= code_bytes * BLOCK && luts.len() >= code_bytes * 32);
        match self {
            Kernel::Scalar => scalar::block_sums(block, luts, code_bytes),
            // SAFETY: Avx2 is only constructed by detect() after
            // is_x86_feature_detected!("avx2"), and the slice lengths
            // the kernel reads were asserted above
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { avx2::block_sums(block, luts, code_bytes) },
            // SAFETY: Neon is only constructed after NEON detection, and
            // the slice lengths were asserted above
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { neon::block_sums(block, luts, code_bytes) },
        }
    }
}

/// Interleaved codes and correction factors of every stored vector
#[derive(Debug, Clone)]
pub(crate) struct FastScanCodes {
    code_bytes: usize,
    blocks: Vec<u8>,
    factors: Vec<f32>,
}

impl FastScanCodes {
    pub(crate) fn new(code_bytes: usize) -> Self {
        Self {
            code_bytes,
            blocks: Vec::new(),
            factors: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.factors.len()
    }

    /// Bytes of one block
    pub(crate) fn block_bytes(&self) -> usize {
        self.code_bytes * BLOCK
    }

    /// Append records (`FACTOR_BYTES` + codes each, `record_size` long)
    pub(crate) fn extend(&mut self, records: &[u8], record_size: usize) {
        let block_bytes = self.block_bytes();
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
            self.factors.push(f32::from_le_bytes([
                factor[0], factor[1], factor[2], factor[3],
            ]));
        }
    }

    /// Keep the first `n` vectors
    pub(crate) fn truncate(&mut self, n: usize) {
        if n >= self.len() {
            return;
        }
        self.factors.truncate(n);
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
