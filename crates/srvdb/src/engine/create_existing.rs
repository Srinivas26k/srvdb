//! `SrvDB::create` with `exist_ok`: open a matching database (ADR 0013)

use super::create_options::CreateOptions;
use super::*;
use crate::storage::settings::{HnswParams, IndexKind};

/// The recorded value of option `name`, if the stored kind has it
fn recorded(kind: &IndexKind, name: &str) -> Option<u64> {
    let graph = |p: &HnswParams| match name {
        "m" => Some(p.m as u64),
        "ef_construction" => Some(p.ef_construction as u64),
        "ef_search" => Some(p.ef_search as u64),
        _ => None,
    };
    match kind {
        IndexKind::Hnsw(p) | IndexKind::HnswPq(p) => graph(p),
        IndexKind::Ivf {
            nlist,
            nprobe,
            hnsw,
        } => match name {
            "nlist" => Some(*nlist as u64),
            "nprobe" => Some(*nprobe as u64),
            _ => graph(hnsw),
        },
        IndexKind::Rabitq(c) => match name {
            "bits" => Some(u64::from(c.bits)),
            "rerank" => Some(c.rerank as u64),
            "seed" => Some(c.seed),
            _ => None,
        },
        _ => None,
    }
}

impl SrvDB {
    /// Open the database at `path` and check that its recorded settings
    /// are the ones `create` was asked for: same index, same dimension and
    /// the same value for every option given. Training vectors are not
    /// compared; an existing quantizer is kept.
    ///
    /// # Errors
    /// `SettingsMismatch` or `DimensionMismatch` naming the difference
    /// (the database is closed again), or any `open_existing` error.
    pub(super) fn open_matching(
        path: &str,
        dimension: usize,
        options: &CreateOptions,
    ) -> Result<Self> {
        let db = Self::open_existing(path)?;
        if db.dimension != dimension {
            return Err(Error::DimensionMismatch {
                expected: db.dimension,
                actual: dimension,
            });
        }
        let index = options.index();
        let kind = db.current_settings().kind;
        let stored = kind.name();
        let pq_with_graph = stored == "hnsw+pq" && index.name() == "pq";
        if stored != index.name() && !pq_with_graph {
            return Err(Error::SettingsMismatch(format!(
                "{path} was created with index '{stored}', not '{index}'; open it with \
                 open_existing (srvdb.open in Python) or delete it first"
            )));
        }
        for name in index.options() {
            let Some(wanted) = options.get(name) else {
                continue;
            };
            let have = recorded(&kind, name);
            if have != Some(wanted) {
                return Err(Error::SettingsMismatch(format!(
                    "{path} was created with {name} = {}, not {wanted}; open it with \
                     open_existing (srvdb.open in Python) or delete it first",
                    have.map_or_else(|| "unset".to_string(), |v| v.to_string())
                )));
            }
        }
        Ok(db)
    }
}
