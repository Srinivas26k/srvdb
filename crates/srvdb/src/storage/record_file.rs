//! Fixed-size record file: header + `count` records, read through a mmap
//!
//! Shared by the flat (f32), SQ8 (u8) and PQ (u8 code) storages. Crash
//! rules (ADR 0003):
//! - records are written at the offset implied by `count`, never "at the
//!   end of the file", so stray bytes from a crash cannot shift ids
//! - `truncate_to` drops records beyond the committed count on open
//! - `sync` writes the header and fsyncs before the commit is recorded
//!
//! The file grows in chunks and stays mapped, so a record is readable as
//! soon as `append` returns.

use super::header::VectorHeader;
use crate::error::Error;
use anyhow::{Context, Result};
use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Minimum growth step, in records
const MIN_GROWTH_RECORDS: u64 = 1024;

pub(crate) struct RecordFile {
    path: PathBuf,
    file: File,
    mmap: Option<Mmap>,
    header: VectorHeader,
    record_size: usize,
    count: u64,
    /// Count as of the last sync (what the header on disk says)
    synced: u64,
    /// Bytes allocated in the file (>= header + count * record_size)
    capacity: u64,
}

impl RecordFile {
    /// Open `path`, creating it with a fresh header if it does not exist.
    ///
    /// An existing file must have the expected dimension and quantization
    /// mode. `count` starts at the header count clamped to the records
    /// actually present; callers then `truncate_to` the committed count.
    pub(crate) fn open(
        path: &Path,
        dimension: usize,
        record_size: usize,
        mode: u8,
    ) -> Result<Self> {
        let exists = path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;

        let header = if exists && file.metadata()?.len() > 0 {
            let mut bytes = [0u8; VectorHeader::SIZE];
            std::io::Read::read_exact(&mut file, &mut bytes).map_err(|_| {
                Error::Corrupt(format!("{} has a truncated header", path.display()))
            })?;
            let header = VectorHeader::from_bytes(&bytes)?;
            if header.dimension as usize != dimension || header.quantization_mode != mode {
                return Err(Error::invalid_input(format!(
                    "{} holds dimension {} (mode {}), but the database was opened with \
                     dimension {dimension} (mode {mode}); the dimension cannot change",
                    path.display(),
                    header.dimension,
                    header.quantization_mode
                ))
                .into());
            }
            header
        } else {
            let header = VectorHeader::new_quantized(dimension, mode)?;
            file.write_all(&header.to_bytes())?;
            file.sync_all()?;
            header
        };

        let capacity = file.metadata()?.len();
        let present = capacity.saturating_sub(VectorHeader::SIZE as u64) / record_size as u64;
        let mut this = Self {
            path: path.to_path_buf(),
            file,
            mmap: None,
            header,
            record_size,
            count: header.count.min(present),
            synced: header.count.min(present),
            capacity,
        };
        this.remap()?;
        Ok(this)
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }

    pub(crate) fn record_size(&self) -> usize {
        self.record_size
    }

    fn offset(&self, index: u64) -> u64 {
        VectorHeader::SIZE as u64 + index * self.record_size as u64
    }

    /// Append whole records; returns the id of the first one.
    pub(crate) fn append(&mut self, bytes: &[u8]) -> Result<u64> {
        if !bytes.len().is_multiple_of(self.record_size) {
            return Err(Error::Internal(format!(
                "append of {} bytes is not a multiple of the {}-byte record size",
                bytes.len(),
                self.record_size
            ))
            .into());
        }
        let first = self.count;
        let added = (bytes.len() / self.record_size) as u64;
        let end = self.offset(first + added);
        if end > self.capacity {
            let step = (self.capacity / 2).max(MIN_GROWTH_RECORDS * self.record_size as u64);
            self.file.set_len(end.max(self.capacity + step))?;
            self.capacity = self.file.metadata()?.len();
            self.remap()?;
        }
        self.file.seek(SeekFrom::Start(self.offset(first)))?;
        self.file.write_all(bytes)?;
        self.count += added;
        Ok(first)
    }

    /// One record's bytes, if `index` is below the count.
    pub(crate) fn get(&self, index: u64) -> Option<&[u8]> {
        self.range(index, 1)
    }

    /// `n` consecutive records starting at `start`.
    pub(crate) fn range(&self, start: u64, n: usize) -> Option<&[u8]> {
        if start.checked_add(n as u64)? > self.count {
            return None;
        }
        let from = self.offset(start) as usize;
        let to = from + n * self.record_size;
        self.mmap.as_ref()?.get(from..to)
    }

    /// Write the header count and fsync the file: everything appended so
    /// far is on disk when this returns.
    pub(crate) fn sync(&mut self) -> Result<()> {
        if self.count == self.synced {
            return Ok(());
        }
        self.header.count = self.count;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file
            .sync_all()
            .with_context(|| format!("failed to fsync {}", self.path.display()))?;
        self.synced = self.count;
        Ok(())
    }

    /// Drop every record from `count` on, shrinking the file. Used on open
    /// to discard writes that were never committed.
    pub(crate) fn truncate_to(&mut self, count: u64) -> Result<()> {
        if count > self.count {
            return Err(Error::Corrupt(format!(
                "{} holds {} records but {count} were committed",
                self.path.display(),
                self.count
            ))
            .into());
        }
        let len = self.offset(count);
        if count == self.count && len == self.capacity {
            return Ok(());
        }
        self.mmap = None;
        self.file.set_len(len)?;
        self.capacity = len;
        self.count = count;
        self.sync()?;
        self.remap()
    }

    fn remap(&mut self) -> Result<()> {
        self.mmap = None;
        if self.capacity > VectorHeader::SIZE as u64 {
            // SAFETY: SrvDB holds the only writer for this file (one database
            // per directory), it only grows or is truncated while unmapped,
            // and reads are bounds-checked against `count`, so mapped bytes
            // are never read while being resized.
            self.mmap = Some(unsafe { MmapOptions::new().map(&self.file)? });
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "record_file_tests.rs"]
mod tests;
