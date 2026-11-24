//! RaBitQ code storage (ADR 0008): derived data, rebuilt from `vectors.bin`
//!
//! Files:
//! - `rabitq_params.bin`: dimension, bit width, rotation seed and rotation
//!   fingerprint, in the checksummed envelope (`storage::format`, kind
//!   `RBQP`), written atomically.
//! - `rabitq_codes.bin`: one record per vector through [`RecordFile`]
//!   (versioned header, CRC32C block sidecar `.crc`, ADR 0007). The header
//!   mode byte is `0x10 | bits`, so a file of another bit width is refused.
//!
//! [`RabitqStorage::open`] fails on anything that does not match the
//! requested settings or its checksums; the engine then calls
//! [`RabitqStorage::create`] and re-encodes from the full vectors.

use super::atomic::write_atomic;
use super::checksum::sidecar;
use super::format;
use super::record_file::RecordFile;
use crate::error::Error;
use crate::quant::rabitq::fastscan::{FastScanCodes, Kernel, QueryLut, BLOCK};
use crate::quant::rabitq::{RabitqConfig, RabitqQuantizer};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

pub const FILE_NAME: &str = "rabitq_codes.bin";
pub const PARAMS_FILE: &str = "rabitq_params.bin";
/// Every file this storage writes (for compaction cleanup)
pub(crate) const ALL_FILES: [&str; 3] = [FILE_NAME, "rabitq_codes.bin.crc", PARAMS_FILE];

/// Payload of `rabitq_params.bin`, format version 1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Params {
    dimension: u32,
    bits: u8,
    seed: u64,
    fingerprint: u32,
}

pub(crate) struct RabitqStorage {
    file: RecordFile,
    quantizer: RabitqQuantizer,
    /// Block-interleaved in-memory copy of the codes the scan reads
    fast: FastScanCodes,
}

fn mode(bits: u8) -> u8 {
    0x10 | bits
}

impl RabitqStorage {
    fn params(dimension: usize, config: &RabitqConfig, q: &RabitqQuantizer) -> Params {
        Params {
            dimension: dimension as u32,
            bits: config.bits,
            seed: config.seed,
            fingerprint: q.rotation_fingerprint(),
        }
    }

    /// Open existing codes made with exactly these settings.
    ///
    /// # Errors
    /// `InvalidState` if there are no codes or they were made with other
    /// settings, `Corrupt` / `ChecksumMismatch` if a file is damaged.
    pub(crate) fn open(dir: &Path, dimension: usize, config: &RabitqConfig) -> Result<Self> {
        let params_path = dir.join(PARAMS_FILE);
        if !params_path.exists() {
            return Err(Error::invalid_state("no RaBitQ codes yet").into());
        }
        let stored: Params = format::decode(
            format::RABITQ_PARAMS,
            &std::fs::read(&params_path)
                .with_context(|| format!("failed to read {PARAMS_FILE}"))?,
        )?;
        let quantizer = RabitqQuantizer::new(dimension, config.bits, config.seed);
        if stored != Self::params(dimension, config, &quantizer) {
            return Err(Error::invalid_state(format!(
                "RaBitQ codes were made with other settings ({} bits, seed {:#x})",
                stored.bits, stored.seed
            ))
            .into());
        }
        let file = RecordFile::open(
            &dir.join(FILE_NAME),
            dimension,
            quantizer.record_size(),
            mode(config.bits),
        )?;
        Self::with_file(file, quantizer)
    }

    /// Wrap an open code file and build its fast-scan copy
    fn with_file(file: RecordFile, quantizer: RabitqQuantizer) -> Result<Self> {
        let mut fast = FastScanCodes::new(quantizer.code_bytes());
        let count = file.count() as usize;
        if count > 0 {
            let records = file.range(0, count).ok_or_else(|| {
                Error::Internal(format!("{FILE_NAME}: {count} records are not mapped"))
            })?;
            fast.extend(records, quantizer.record_size())?;
        }
        Ok(Self {
            file,
            quantizer,
            fast,
        })
    }

    /// Delete any existing codes and start empty ones with these settings.
    pub(crate) fn create(dir: &Path, dimension: usize, config: &RabitqConfig) -> Result<Self> {
        Self::remove(dir)?;
        let quantizer = RabitqQuantizer::new(dimension, config.bits, config.seed);
        let params = Self::params(dimension, config, &quantizer);
        write_atomic(
            &dir.join(PARAMS_FILE),
            &format::encode(format::RABITQ_PARAMS, &params)?,
        )?;
        let file = RecordFile::open(
            &dir.join(FILE_NAME),
            dimension,
            quantizer.record_size(),
            mode(config.bits),
        )?;
        Self::with_file(file, quantizer)
    }

    /// Delete the code files (codes first, so a crash leaves no codes
    /// next to parameters that would claim them).
    pub(crate) fn remove(dir: &Path) -> Result<()> {
        let codes = dir.join(FILE_NAME);
        for path in [codes.clone(), sidecar(&codes), dir.join(PARAMS_FILE)] {
            if path.exists() {
                std::fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
            }
        }
        Ok(())
    }

    /// Encode and append vectors; ids continue from [`Self::count`].
    /// Encoding (one O(D^2) rotation per vector) runs on the rayon pool.
    pub(crate) fn append<'a>(&mut self, vectors: impl Iterator<Item = &'a [f32]>) -> Result<()> {
        use rayon::prelude::*;
        let vectors: Vec<&[f32]> = vectors.collect();
        let size = self.quantizer.record_size();
        let mut bytes = vec![0u8; vectors.len() * size];
        bytes
            .par_chunks_mut(size)
            .zip(vectors.par_iter())
            .for_each(|(out, v)| {
                let mut record = Vec::with_capacity(size);
                self.quantizer.encode_into(v, &mut record);
                out.copy_from_slice(&record);
            });
        if !bytes.is_empty() {
            // Copy first, so a failed reservation leaves both unchanged;
            // a failed file append rolls the copy back
            let before = self.fast.len();
            self.fast.extend(&bytes, size)?;
            if let Err(e) = self.file.append(&bytes) {
                self.fast.truncate(before);
                return Err(e);
            }
        }
        Ok(())
    }

    /// The `n` best ids by estimated cosine similarity, skipping
    /// `deleted`. Runs of blocks are scanned in parallel (rayon) with the
    /// fastest kernel this CPU supports.
    pub(crate) fn search(
        &self,
        query: &[f32],
        n: usize,
        deleted: &HashSet<u64>,
    ) -> Result<Vec<(u64, f32)>> {
        use rayon::prelude::*;
        /// Blocks per parallel task (1024 vectors)
        const TASK_BLOCKS: usize = 32;
        if self.fast.len() == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let lut = QueryLut::new(self.quantizer.prepare(query).tables());
        let kernel = Kernel::detect();
        let blocks = self.fast.len().div_ceil(BLOCK);
        let tasks = blocks.div_ceil(TASK_BLOCKS);
        let scored: Vec<(u64, f32)> = (0..tasks)
            .into_par_iter()
            .flat_map_iter(|task| {
                let mut out = Vec::with_capacity(TASK_BLOCKS * BLOCK);
                let mut scores = [0.0f32; BLOCK];
                let end = ((task + 1) * TASK_BLOCKS).min(blocks);
                for b in task * TASK_BLOCKS..end {
                    let real = self.fast.block_scores(b, &lut, kernel, &mut scores);
                    let first = (b * BLOCK) as u64;
                    out.extend(
                        scores[..real]
                            .iter()
                            .zip(first..)
                            .filter(|(_, id)| deleted.is_empty() || !deleted.contains(id))
                            .map(|(s, id)| (id, *s)),
                    );
                }
                // Each task keeps only its own best n
                crate::distance::select_top_k(out, n)
            })
            .collect();
        Ok(crate::distance::select_top_k(scored, n))
    }

    pub(crate) fn count(&self) -> u64 {
        self.file.count()
    }

    /// Make appended codes durable (fsync + checksums)
    pub(crate) fn flush(&mut self) -> Result<()> {
        self.file.sync()
    }

    /// Drop codes from `count` on
    pub(crate) fn truncate_to(&mut self, count: u64) -> Result<()> {
        self.file.truncate_to(count)?;
        self.fast.truncate(count as usize);
        Ok(())
    }
}

#[cfg(test)]
#[path = "rabitq_tests.rs"]
mod tests;
