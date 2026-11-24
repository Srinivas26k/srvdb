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
use crate::quant::rabitq::{RabitqConfig, RabitqQuantizer, RabitqQuery};
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
        Ok(Self { file, quantizer })
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
        Ok(Self { file, quantizer })
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
            self.file.append(&bytes)?;
        }
        Ok(())
    }

    /// The `n` best ids by estimated cosine similarity, skipping
    /// `deleted`. Blocks of records are scored in parallel (rayon).
    pub(crate) fn search(
        &self,
        query: &[f32],
        n: usize,
        deleted: &HashSet<u64>,
    ) -> Vec<(u64, f32)> {
        use crate::distance::BATCH_SIZE;
        use rayon::prelude::*;
        let size = self.file.record_size();
        let Some(records) = self.file.range(0, self.file.count() as usize) else {
            return Vec::new();
        };
        let prepared: RabitqQuery = self.quantizer.prepare(query);
        let scored: Vec<(u64, f32)> = records
            .par_chunks(BATCH_SIZE * size)
            .enumerate()
            .flat_map_iter(|(batch, block)| {
                let first = (batch * BATCH_SIZE) as u64;
                let prepared = &prepared;
                block
                    .chunks_exact(size)
                    .zip(first..)
                    .filter(|(_, id)| !deleted.contains(id))
                    .map(move |(r, id)| (id, prepared.score(r)))
            })
            .collect();
        crate::distance::select_top_k(scored, n)
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
        self.file.truncate_to(count)
    }
}

#[cfg(test)]
#[path = "rabitq_tests.rs"]
mod tests;
