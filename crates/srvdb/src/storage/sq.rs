//! Scalar-quantized (SQ8) storage: one u8 per dimension, 4x smaller than f32
//!
//! Files: `sq8_vectors.bin` (codes, via [`RecordFile`]) and
//! `scalar_quantizer.bin` (per-dimension ranges, written atomically).

use super::atomic::write_atomic;
use super::format;
use super::record_file::RecordFile;
use crate::error::Error;
use crate::quant::scalar::ScalarQuantizer;
use anyhow::{Context, Result};
use std::path::Path;

pub const FILE_NAME: &str = "sq8_vectors.bin";
pub const QUANTIZER_FILE: &str = "scalar_quantizer.bin";
const MODE_SQ8: u8 = 1;

pub struct ScalarQuantizedStorage {
    file: RecordFile,
    dimension: usize,
    quantizer: ScalarQuantizer,
}

impl ScalarQuantizedStorage {
    /// Train a quantizer on `training_data` and create empty SQ8 storage.
    ///
    /// Refuses to run on a directory that already holds SQ8 vectors: they
    /// were encoded with the existing quantizer, so retraining would make
    /// them unreadable. Open such a database with [`Self::load`].
    pub fn new_with_training(
        db_path: &str,
        dimension: usize,
        training_data: &[Vec<f32>],
    ) -> Result<Self> {
        let dir = Path::new(db_path);
        let existing = RecordFile::open(&dir.join(FILE_NAME), dimension, dimension, MODE_SQ8)?;
        if existing.count() > 0 {
            return Err(Error::invalid_state(format!(
                "{db_path} already holds {} SQ8 vectors; open it instead of training a new quantizer",
                existing.count()
            ))
            .into());
        }

        let quantizer = ScalarQuantizer::train(training_data)?;
        write_atomic(
            &dir.join(QUANTIZER_FILE),
            &format::encode(format::SQ8_QUANTIZER, &quantizer)?,
        )?;
        Ok(Self {
            file: existing,
            dimension,
            quantizer,
        })
    }

    /// Open existing SQ8 storage
    pub fn load(db_path: &str, dimension: usize) -> Result<Self> {
        let dir = Path::new(db_path);
        let bytes = std::fs::read(dir.join(QUANTIZER_FILE))
            .with_context(|| format!("failed to read {QUANTIZER_FILE}"))?;
        let quantizer: ScalarQuantizer = if format::is_enveloped(&bytes) {
            format::decode(format::SQ8_QUANTIZER, &bytes)?
        } else {
            super::legacy::scalar_quantizer(&bytes)?
        };
        Ok(Self {
            file: RecordFile::open(&dir.join(FILE_NAME), dimension, dimension, MODE_SQ8)?,
            dimension,
            quantizer,
        })
    }

    fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            }
            .into());
        }
        Ok(self.quantizer.encode(vector))
    }

    /// Quantize and append one vector; searchable as soon as this returns.
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        let codes = self.encode(vector)?;
        self.file.append(&codes)
    }

    /// Quantize and append vectors in one write; returns their ids.
    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let mut codes = Vec::with_capacity(vectors.len() * self.dimension);
        for vector in vectors {
            codes.extend(self.encode(vector)?);
        }
        let first = self.file.append(&codes)?;
        Ok((first..first + vectors.len() as u64).collect())
    }

    /// Decode one stored vector (approximate)
    pub fn get(&self, index: u64) -> Option<Vec<f32>> {
        Some(self.quantizer.decode(self.file.get(index)?))
    }

    /// Exhaustive search comparing the full-precision query to every code
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        let count = self.file.count();
        let Some(codes) = self.file.range(0, count as usize) else {
            return Ok(Vec::new());
        };
        let mut results: Vec<(u64, f32)> = codes
            .chunks_exact(self.dimension)
            .enumerate()
            .map(|(i, code)| (i as u64, self.quantizer.asymmetric_distance(query, code)))
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);
        Ok(results)
    }

    pub fn count(&self) -> u64 {
        self.file.count()
    }

    /// Make everything appended so far durable (fsync)
    pub fn flush(&mut self) -> Result<()> {
        self.file.sync()
    }

    /// Discard vectors from `count` on (uncommitted after a crash)
    pub(crate) fn truncate_to(&mut self, count: u64) -> Result<()> {
        self.file.truncate_to(count)
    }
}
