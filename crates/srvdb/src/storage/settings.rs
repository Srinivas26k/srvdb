//! Persisted index settings (ADR 0011): `index_settings.bin`
//!
//! A small manifest that records which index mode a database was created
//! with and the parameters that mode needs to reopen. It lives in the
//! checksummed envelope of `storage/format.rs` (kind `IDXS`, CRC32C) and is
//! replaced atomically (temp file, fsync, rename, directory fsync).
//!
//! The payload is a fixed set of little-endian integers (postcard), with a
//! payload `format` number of its own so fields can be added later. Unused
//! fields must be zero. Everything is validated on load; an unknown kind or
//! format is `Error::Corrupt` naming the problem, a damaged payload is
//! `Error::ChecksumMismatch`.

use super::{atomic::write_atomic, format};
use crate::error::Error;
use crate::quant::rabitq::RabitqConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// File name inside the database directory
pub const FILE_NAME: &str = "index_settings.bin";
/// Payload format written by this version
const FORMAT: u16 = 1;
/// Upper bound for dimensions and graph/partition parameters read from disk
const MAX_PARAM: u64 = 1_000_000;

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
}

/// On-disk payload; see the module docs
#[derive(Debug, Default, Serialize, Deserialize)]
struct Record {
    format: u16,
    kind: u8,
    dimension: u64,
    m: u64,
    ef_construction: u64,
    ef_search: u64,
    nlist: u64,
    nprobe: u64,
    bits: u8,
    rerank: u64,
    seed: u64,
}

fn hnsw_fields(record: &mut Record, p: &HnswParams) {
    record.m = p.m as u64;
    record.ef_construction = p.ef_construction as u64;
    record.ef_search = p.ef_search as u64;
}

fn to_record(s: &IndexSettings) -> Record {
    let mut r = Record {
        format: FORMAT,
        dimension: s.dimension as u64,
        ..Default::default()
    };
    r.kind = match &s.kind {
        IndexKind::Flat => 0,
        IndexKind::Hnsw(p) => {
            hnsw_fields(&mut r, p);
            1
        }
        IndexKind::Sq8 => 2,
        IndexKind::Pq => 3,
        IndexKind::HnswPq(p) => {
            hnsw_fields(&mut r, p);
            4
        }
        IndexKind::Ivf {
            nlist,
            nprobe,
            hnsw,
        } => {
            hnsw_fields(&mut r, hnsw);
            r.nlist = *nlist as u64;
            r.nprobe = *nprobe as u64;
            5
        }
        IndexKind::Rabitq(c) => {
            r.bits = c.bits;
            r.rerank = c.rerank as u64;
            r.seed = c.seed;
            6
        }
        IndexKind::Auto => 7,
    };
    r
}

fn corrupt(why: impl std::fmt::Display) -> anyhow::Error {
    Error::Corrupt(format!(
        "{FILE_NAME}: {why}; restore it from a backup, or delete it and open the \
         database once with its index mode to record the settings again"
    ))
    .into()
}

/// A parameter in `1..=max`
fn param(name: &str, value: u64, max: u64) -> Result<usize> {
    if value == 0 || value > max {
        return Err(corrupt(format!("{name} = {value} is out of range")));
    }
    Ok(value as usize)
}

fn hnsw_params(r: &Record) -> Result<HnswParams> {
    Ok(HnswParams {
        m: param("m", r.m, 4096)?,
        ef_construction: param("ef_construction", r.ef_construction, MAX_PARAM)?,
        ef_search: param("ef_search", r.ef_search, MAX_PARAM)?,
    })
}

fn from_record(r: &Record) -> Result<IndexSettings> {
    if r.format != FORMAT {
        return Err(corrupt(format!(
            "settings format {} is not supported by this SrvDB (expects {FORMAT}); \
             upgrade SrvDB",
            r.format
        )));
    }
    let dimension = param("dimension", r.dimension, 65_535)?;
    let hnsw_used = (r.m | r.ef_construction | r.ef_search) != 0;
    let ivf_used = (r.nlist | r.nprobe) != 0;
    let rabitq_used = r.bits != 0 || r.rerank != 0 || r.seed != 0;
    let (kind, uses_hnsw, uses_ivf, uses_rabitq) = match r.kind {
        0 => (IndexKind::Flat, false, false, false),
        1 => (IndexKind::Hnsw(hnsw_params(r)?), true, false, false),
        2 => (IndexKind::Sq8, false, false, false),
        3 => (IndexKind::Pq, false, false, false),
        4 => (IndexKind::HnswPq(hnsw_params(r)?), true, false, false),
        5 => {
            let nlist = param("nlist", r.nlist, MAX_PARAM)?;
            let nprobe = param("nprobe", r.nprobe, r.nlist)?;
            let hnsw = hnsw_params(r)?;
            (
                IndexKind::Ivf {
                    nlist,
                    nprobe,
                    hnsw,
                },
                true,
                true,
                false,
            )
        }
        6 => {
            let config = RabitqConfig::new(r.bits)
                .map_err(|e| corrupt(e.to_string()))?
                .with_rerank(r.rerank as usize)
                .with_seed(r.seed);
            config.validate().map_err(|e| corrupt(e.to_string()))?;
            (IndexKind::Rabitq(config), false, false, true)
        }
        7 => (IndexKind::Auto, false, false, false),
        other => {
            return Err(corrupt(format!(
                "unknown index kind {other} (written by a newer SrvDB?); upgrade SrvDB"
            )))
        }
    };
    if (hnsw_used && !uses_hnsw) || (ivf_used && !uses_ivf) || (rabitq_used && !uses_rabitq) {
        return Err(corrupt(format!(
            "parameters set that the {} mode does not use",
            kind.name()
        )));
    }
    Ok(IndexSettings { dimension, kind })
}

/// Encode settings in the checksummed envelope
pub(crate) fn encode(settings: &IndexSettings) -> Result<Vec<u8>> {
    format::encode(format::INDEX_SETTINGS, &to_record(settings))
}

/// Decode and validate an `index_settings.bin` image
pub(crate) fn decode(bytes: &[u8]) -> Result<IndexSettings> {
    let record: Record = format::decode(format::INDEX_SETTINGS, bytes)?;
    from_record(&record)
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
