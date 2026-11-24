//! Persisted index settings (ADR 0011): `index_settings.bin`
//!
//! A small manifest that records which index mode a database was created
//! with and the parameters that mode needs to reopen. It lives in the
//! checksummed envelope of `storage/format.rs` (kind `IDXS`, CRC32C) and is
//! replaced atomically (temp file, fsync, rename, directory fsync).
//!
//! The payload is a fixed set of little-endian integers (postcard), with a
//! payload `format` number of its own so fields can be added later. Unused
//! fields must be zero. Format 2 (ADR 0018) appends `memory_mb`; it is
//! written only for a database with a memory budget, so databases without
//! one stay readable by builds that know format 1 only. Everything is validated on load; an unknown kind or
//! format is `Error::Corrupt` naming the problem, a damaged payload is
//! `Error::ChecksumMismatch`.

use super::{atomic::write_atomic, format};
use crate::quant::rabitq::RabitqConfig;
use anyhow::{Context, Result};
use std::path::Path;

/// File name inside the database directory
pub const FILE_NAME: &str = "index_settings.bin";
/// Payload format without a memory budget (ADR 0011)
const FORMAT_V1: u16 = 1;
/// Payload format with a memory budget (ADR 0018)
const FORMAT_V2: u16 = 2;

/// HNSW parameters that shape a graph (`m`, `ef_construction`) and the
/// search-time candidate list (`ef_search`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

/// The index mode of a database and its parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexKind {
    Flat,
    Hnsw(HnswParams),
    Sq8,
    /// PQ parameters live in the checksummed `codebooks.bin`
    Pq,
    HnswPq(HnswParams),
    Ivf {
        nlist: usize,
        nprobe: usize,
        hnsw: HnswParams,
    },
    Rabitq(RabitqConfig),
    Auto,
}

impl IndexKind {
    /// Mode name used in error messages
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Hnsw(_) => "hnsw",
            Self::Sq8 => "sq8",
            Self::Pq => "pq",
            Self::HnswPq(_) => "hnsw+pq",
            Self::Ivf { .. } => "ivf",
            Self::Rabitq(_) => "rabitq",
            Self::Auto => "auto",
        }
    }
}

/// What `index_settings.bin` holds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexSettings {
    pub dimension: usize,
    pub kind: IndexKind,
    /// Memory budget in MiB (ADR 0018); `kind` is flat or RaBitQ with it
    pub memory_mb: Option<u64>,
}

#[path = "settings_record.rs"]
mod record;
use record::{corrupt, from_record, to_record, Record, RecordV2};

/// Encode settings in the checksummed envelope: format 1 without a
/// memory budget, format 2 with one
pub(crate) fn encode(settings: &IndexSettings) -> Result<Vec<u8>> {
    let base = to_record(settings);
    match settings.memory_mb {
        None => format::encode(format::INDEX_SETTINGS, &base),
        Some(memory_mb) => format::encode(
            format::INDEX_SETTINGS,
            &RecordV2 {
                base: Record {
                    format: FORMAT_V2,
                    ..base
                },
                memory_mb,
            },
        ),
    }
}

/// Decode and validate an `index_settings.bin` image of format 1 or 2
pub(crate) fn decode(bytes: &[u8]) -> Result<IndexSettings> {
    // The format number comes first; read it, then the whole payload of
    // that format, refusing bytes after it
    let head: Record = format::decode(format::INDEX_SETTINGS, bytes)?;
    if head.format == FORMAT_V1 {
        let record: Record = format::decode_exact(format::INDEX_SETTINGS, bytes)?;
        return from_record(&record, 0);
    }
    if head.format != FORMAT_V2 {
        // Unknown format: from_record names it and says to upgrade
        return from_record(&head, 0);
    }
    let v2: RecordV2 = format::decode_exact(format::INDEX_SETTINGS, bytes)?;
    if v2.memory_mb == 0 {
        return Err(corrupt("format 2 without a memory budget"));
    }
    from_record(&v2.base, v2.memory_mb)
}

/// Replace `index_settings.bin` in `dir` atomically
pub(crate) fn save(dir: &Path, settings: &IndexSettings) -> Result<()> {
    write_atomic(&dir.join(FILE_NAME), &encode(settings)?)
}

/// Read `index_settings.bin` from `dir`; `None` if the database has none
/// (created by SrvDB 0.3 or an earlier 0.4 build, or not created yet).
pub(crate) fn load(dir: &Path) -> Result<Option<IndexSettings>> {
    let path = dir.join(FILE_NAME);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("failed to read {}", path.display())),
    };
    decode(&bytes).map(Some)
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
