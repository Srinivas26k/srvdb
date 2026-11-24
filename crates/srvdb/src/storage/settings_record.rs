//! The `index_settings.bin` payload and its validation (ADR 0011, 0018)

use super::{HnswParams, IndexKind, IndexSettings, FILE_NAME, FORMAT_V1, FORMAT_V2};
use crate::error::Error;
use crate::index::limits;
use crate::quant::rabitq::RabitqConfig;
use crate::tuning::budget::MAX_MEMORY_MB;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// On-disk payload; see the module docs. Format 1 is exactly these
/// fields; format 2 is [`RecordV2`].
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(super) struct Record {
    pub(super) format: u16,
    pub(super) kind: u8,
    pub(super) dimension: u64,
    pub(super) m: u64,
    pub(super) ef_construction: u64,
    pub(super) ef_search: u64,
    pub(super) nlist: u64,
    pub(super) nprobe: u64,
    pub(super) bits: u8,
    pub(super) rerank: u64,
    pub(super) seed: u64,
}

/// Format 2: format 1 followed by the memory budget (0 = none)
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct RecordV2 {
    pub(super) base: Record,
    pub(super) memory_mb: u64,
}

fn hnsw_fields(record: &mut Record, p: &HnswParams) {
    record.m = p.m as u64;
    record.ef_construction = p.ef_construction as u64;
    record.ef_search = p.ef_search as u64;
}

pub(super) fn to_record(s: &IndexSettings) -> Record {
    let mut r = Record {
        format: FORMAT_V1,
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

pub(super) fn corrupt(why: impl std::fmt::Display) -> anyhow::Error {
    Error::Corrupt(format!(
        "{FILE_NAME}: {why}; restore it from a backup, or delete it and open the \
         database once with its index mode to record the settings again"
    ))
    .into()
}

/// A parameter in `min..=max`: the same bounds the public API enforces
/// (`index::limits`), checked before anything is allocated from it
fn param(name: &str, value: u64, min: usize, max: usize) -> Result<usize> {
    if value < min as u64 || value > max as u64 {
        return Err(corrupt(format!(
            "{name} = {value} is outside {min}..={max}"
        )));
    }
    Ok(value as usize)
}

fn hnsw_params(r: &Record) -> Result<HnswParams> {
    Ok(HnswParams {
        m: param("m", r.m, limits::MIN_HNSW_M, limits::MAX_HNSW_M)?,
        ef_construction: param("ef_construction", r.ef_construction, 1, limits::MAX_HNSW_EF)?,
        ef_search: param("ef_search", r.ef_search, 1, limits::MAX_HNSW_EF)?,
    })
}

pub(super) fn from_record(r: &Record, memory_mb: u64) -> Result<IndexSettings> {
    if r.format != FORMAT_V1 && r.format != FORMAT_V2 {
        return Err(corrupt(format!(
            "settings format {} is not supported by this SrvDB (expects \
             {FORMAT_V1} or {FORMAT_V2}); upgrade SrvDB",
            r.format
        )));
    }
    let dimension = param("dimension", r.dimension, 1, limits::MAX_DIMENSION)?;
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
            let nlist = param("nlist", r.nlist, 1, limits::MAX_IVF_NLIST)?;
            let nprobe = param("nprobe", r.nprobe, 1, limits::MAX_IVF_NLIST)?;
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
    let memory_mb = match memory_mb {
        0 => None,
        mb => {
            let mb = param("memory_mb", mb, 1, MAX_MEMORY_MB as usize)? as u64;
            if !matches!(kind, IndexKind::Flat | IndexKind::Rabitq(_)) {
                return Err(corrupt(format!(
                    "a memory budget is recorded for the {} mode, which cannot have one",
                    kind.name()
                )));
            }
            Some(mb)
        }
    };
    Ok(IndexSettings {
        dimension,
        kind,
        memory_mb,
    })
}
