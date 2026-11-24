//! Block checksums for record files (ADR 0007)
//!
//! The record area of a vector file (everything after the 24-byte header)
//! is split into fixed 1 MiB blocks; each block has a CRC32C. The table
//! lives in a sidecar, `<file>.crc`, written atomically in the checksummed
//! envelope (`storage::format`) and covering exactly `count` records. The
//! last block is partial and its CRC covers only the bytes present.
//!
//! The table is kept up to date in memory as records are appended
//! (`crc32c_append` extends a CRC without rereading earlier bytes) and is
//! written by `RecordFile::sync` after the vector file is fsynced.

use super::atomic::write_atomic;
use super::format;
use crate::error::Error;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bytes covered by one checksum
pub(crate) const BLOCK_BYTES: u64 = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BlockSums {
    block_bytes: u64,
    record_size: u64,
    /// Records covered by the table
    count: u64,
    crcs: Vec<u32>,
}

/// Sidecar path for a record file: `vectors.bin` -> `vectors.bin.crc`
pub(crate) fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".crc");
    PathBuf::from(name)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

impl BlockSums {
    /// An empty table (no records)
    pub(crate) fn new(record_size: usize) -> Self {
        Self {
            block_bytes: BLOCK_BYTES,
            record_size: record_size as u64,
            count: 0,
            crcs: Vec::new(),
        }
    }

    /// Checksum `count` records held in `data` (the record area)
    pub(crate) fn compute(record_size: usize, count: u64, data: &[u8]) -> Result<Self> {
        let mut sums = Self::new(record_size);
        sums.append(&data[..(count * record_size as u64) as usize])?;
        Ok(sums)
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }

    fn data_len(&self) -> u64 {
        self.count * self.record_size
    }

    /// Extend the table with whole records appended after `count`.
    /// A partial record is an internal error and leaves the table as is.
    pub(crate) fn append(&mut self, mut bytes: &[u8]) -> Result<()> {
        if !(bytes.len() as u64).is_multiple_of(self.record_size) {
            return Err(Error::Internal(format!(
                "checksum append of {} bytes is not a multiple of the {}-byte record size",
                bytes.len(),
                self.record_size
            ))
            .into());
        }
        let mut offset = self.data_len();
        self.count += bytes.len() as u64 / self.record_size;
        while !bytes.is_empty() {
            let within = offset % self.block_bytes;
            let take = ((self.block_bytes - within) as usize).min(bytes.len());
            let (chunk, rest) = bytes.split_at(take);
            if within == 0 {
                self.crcs.push(crc32c::crc32c(chunk));
            } else if let Some(last) = self.crcs.last_mut() {
                *last = crc32c::crc32c_append(*last, chunk);
            }
            offset += take as u64;
            bytes = rest;
        }
        Ok(())
    }

    /// Shrink to `count` records; `data` is the record area, already
    /// verified, used to recompute the new partial last block.
    pub(crate) fn truncate(&mut self, count: u64, data: &[u8]) {
        let len = count * self.record_size;
        let blocks = len.div_ceil(self.block_bytes) as usize;
        self.crcs.truncate(blocks);
        if blocks > 0 {
            let start = (blocks as u64 - 1) * self.block_bytes;
            self.crcs[blocks - 1] = crc32c::crc32c(&data[start as usize..len as usize]);
        }
        self.count = count;
    }

    /// Check every block of `data` (the record area of `path`) against the
    /// table. `data` must hold at least `count` records.
    pub(crate) fn verify(&self, path: &Path, data: &[u8]) -> Result<()> {
        let len = self.data_len() as usize;
        let data = data.get(..len).ok_or_else(|| {
            Error::Corrupt(format!(
                "{} is truncated: {} records are checksummed but fewer are present",
                file_name(path),
                self.count
            ))
        })?;
        for (i, (block, crc)) in data
            .chunks(self.block_bytes as usize)
            .zip(&self.crcs)
            .enumerate()
        {
            if crc32c::crc32c(block) != *crc {
                let start = i as u64 * self.block_bytes;
                return Err(Error::ChecksumMismatch(format!(
                    "{} bytes {}..{} after the header do not match their checksum; \
                     the file is damaged, restore the database from a backup",
                    file_name(path),
                    start,
                    start + block.len() as u64
                ))
                .into());
            }
        }
        Ok(())
    }

    /// Write the table atomically to the sidecar of `path`
    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        write_atomic(&sidecar(path), &format::encode(format::CHECKSUMS, self)?)
    }

    /// Decode sidecar bytes (envelope check only, no shape check)
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        format::decode(format::CHECKSUMS, bytes)
    }

    /// Whether the table is internally consistent and is for records of
    /// `record_size` bytes
    pub(crate) fn describes(&self, record_size: usize) -> bool {
        let expected = self
            .count
            .checked_mul(self.record_size)
            .map(|len| len.div_ceil(self.block_bytes.max(1)));
        self.block_bytes != 0
            && self.record_size == record_size as u64
            && expected == Some(self.crcs.len() as u64)
    }

    /// Read the sidecar of `path`, or `None` if it does not exist.
    /// The table must match `record_size` and be internally consistent.
    pub(crate) fn load(path: &Path, record_size: usize) -> Result<Option<Self>> {
        let side = sidecar(path);
        if !side.exists() {
            return Ok(None);
        }
        let bytes =
            std::fs::read(&side).with_context(|| format!("failed to read {}", side.display()))?;
        let sums: Self = format::decode(format::CHECKSUMS, &bytes)?;
        if !sums.describes(record_size) {
            return Err(Error::Corrupt(format!(
                "{} does not describe {} (record size or block count is wrong); \
                 restore the database from a backup",
                file_name(&side),
                file_name(path)
            ))
            .into());
        }
        Ok(Some(sums))
    }
}

#[cfg(test)]
#[path = "checksum_tests.rs"]
mod tests;
