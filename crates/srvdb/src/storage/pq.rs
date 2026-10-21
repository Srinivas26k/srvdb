//! Product-quantized (PQ) storage: `m` one-byte codes per vector
//!
//! Files: `quantized_vectors.bin` (codes, via [`RecordFile`]) and
//! `codebooks.bin` (trained codebooks, written atomically).

use super::atomic::write_atomic;
use super::record_file::RecordFile;
use crate::error::Error;
use crate::quant::product::ProductQuantizer;
use anyhow::{Context, Result};
use std::path::Path;

#[path = "pq_stats.rs"]
mod stats;
pub use stats::StorageStats;

pub const FILE_NAME: &str = "quantized_vectors.bin";
pub const CODEBOOK_FILE: &str = "codebooks.bin";
const MODE_PQ: u8 = 2;

pub struct QuantizedVectorStorage {
    file: RecordFile,
    pub quantizer: ProductQuantizer,
    /// Bytes per stored vector (the number of sub-quantizers, `m`)
    pub q_size: usize,
}

/// Sub-quantizer count for a dimension: 8 dimensions per sub-space when
/// possible, otherwise 4.
fn subquantizers(dim: usize) -> Result<usize> {
    match dim {
        0 => Err(Error::invalid_input("training vectors are empty").into()),
        d if d % 8 == 0 => Ok(d / 8),
        d if d % 4 == 0 => Ok(d / 4),
        d => Err(
            Error::invalid_input(format!("PQ needs a dimension divisible by 4, got {d}")).into(),
        ),
    }
}

impl QuantizedVectorStorage {
    /// Train codebooks on `training_data` and create empty PQ storage.
    ///
    /// Refuses to run on a directory that already holds PQ vectors: they
    /// were encoded with the existing codebooks, so retraining would make
    /// them unreadable. Open such a database with [`Self::load`].
    pub fn new_with_training(db_path: &str, training_data: &[Vec<f32>]) -> Result<Self> {
        let dir = Path::new(db_path);
        let dim = training_data.first().map_or(0, Vec::len);
        let m = subquantizers(dim)?;
        let existing = RecordFile::open(&dir.join(FILE_NAME), dim, m, MODE_PQ)?;
        if existing.count() > 0 {
            return Err(Error::invalid_state(format!(
                "{db_path} already holds {} PQ vectors; open it instead of training new codebooks",
                existing.count()
            ))
            .into());
        }

        let quantizer = ProductQuantizer::train(training_data, m)?;
        log::debug!("PQ training complete");
        write_atomic(&dir.join(CODEBOOK_FILE), &quantizer.to_bytes()?)?;
        Ok(Self {
            file: existing,
            quantizer,
            q_size: m,
        })
    }

    /// Open existing PQ storage
    pub fn load(db_path: &str) -> Result<Self> {
        let dir = Path::new(db_path);
        let bytes = std::fs::read(dir.join(CODEBOOK_FILE))
            .with_context(|| format!("failed to read {CODEBOOK_FILE}"))?;
        let quantizer = ProductQuantizer::from_bytes(&bytes)
            .map_err(|e| Error::Corrupt(format!("{CODEBOOK_FILE}: {e:#}")))?;
        let m = quantizer.m;
        let dim = m * quantizer.d_sub;
        Ok(Self {
            file: RecordFile::open(&dir.join(FILE_NAME), dim, m, MODE_PQ)?,
            quantizer,
            q_size: m,
        })
    }

    fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let expected = self.q_size * self.quantizer.d_sub;
        if vector.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                actual: vector.len(),
            }
            .into());
        }
        Ok(self.quantizer.quantize(vector))
    }

    /// Quantize and append one vector; searchable as soon as this returns.
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        let codes = self.encode(vector)?;
        self.file.append(&codes)
    }

    /// Quantize and append vectors in one write; returns their ids.
    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let mut codes = Vec::with_capacity(vectors.len() * self.q_size);
        for vector in vectors {
            codes.extend(self.encode(vector)?);
        }
        let first = self.file.append(&codes)?;
        Ok((first..first + vectors.len() as u64).collect())
    }

    /// Make everything appended so far durable (fsync)
    pub fn flush(&mut self) -> Result<()> {
        self.file.sync()
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.file.count()
    }

    /// Codes of one vector
    #[inline]
    pub fn get(&self, index: u64) -> Option<&[u8]> {
        self.file.get(index)
    }

    /// Codes of `count` consecutive vectors, concatenated
    pub fn get_batch(&self, start: u64, count: usize) -> Option<&[u8]> {
        self.file.range(start, count)
    }
}

#[cfg(test)]
#[path = "pq_tests.rs"]
mod tests;
