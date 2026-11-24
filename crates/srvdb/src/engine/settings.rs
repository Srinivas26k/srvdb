//! Recording and checking the persisted index settings (ADR 0011)

use super::*;
use crate::storage::settings::{self as stored, HnswParams, IndexKind, IndexSettings};

/// What a constructor asks for, compared against the stored settings
pub(super) enum Requested {
    /// No mode given (default `new_with_config`): use what is stored
    Unspecified,
    /// A mode without parameters (`new_with_config` with an index type or
    /// quantization): only the mode must match
    KindOnly(IndexKind),
    /// A mode with explicit parameters: the mode and its structural
    /// parameters must match; search-time knobs (`ef_search`, `nprobe`,
    /// `rerank`, and every RaBitQ setting, see ADR 0008) may differ and the
    /// caller's value is used and recorded by the next `persist()`
    Exact(IndexKind),
}

pub(super) fn hnsw_params(config: &hnsw::HNSWConfig) -> HnswParams {
    HnswParams {
        m: config.m,
        ef_construction: config.ef_construction,
        ef_search: config.ef_search,
    }
}

pub(super) fn hnsw_config(p: &HnswParams) -> hnsw::HNSWConfig {
    hnsw::HNSWConfig::new(p.m, p.ef_construction, p.ef_search)
}

/// Whether `dir` holds no vector file yet (a database being created)
pub(super) fn is_fresh(dir: &Path) -> bool {
    [
        storage::flat::FILE_NAME,
        storage::sq::FILE_NAME,
        storage::pq::FILE_NAME,
    ]
    .iter()
    .all(|name| !dir.join(name).exists())
}

fn same_kind(stored: &IndexKind, requested: &IndexKind) -> bool {
    use std::mem::discriminant;
    discriminant(stored) == discriminant(requested)
        // new_with_config(PQ) reopens a PQ database with or without a graph
        || (matches!(requested, IndexKind::Pq) && matches!(stored, IndexKind::HnswPq(_)))
}

fn same_structure(stored: &IndexKind, requested: &IndexKind) -> bool {
    let graph =
        |a: &HnswParams, b: &HnswParams| a.m == b.m && a.ef_construction == b.ef_construction;
    match (stored, requested) {
        (IndexKind::Hnsw(a), IndexKind::Hnsw(b)) | (IndexKind::HnswPq(a), IndexKind::HnswPq(b)) => {
            graph(a, b)
        }
        (
            IndexKind::Ivf { nlist, hnsw, .. },
            IndexKind::Ivf {
                nlist: n2,
                hnsw: h2,
                ..
            },
        ) => nlist == n2 && graph(hnsw, h2),
        // RaBitQ codes are derived from the stored vectors: another bit width
        // or seed rebuilds them on open (ADR 0008), so only the mode matters
        (IndexKind::Rabitq(_), IndexKind::Rabitq(_)) => true,
        (a, b) => std::mem::discriminant(a) == std::mem::discriminant(b),
    }
}

/// Read the stored settings of `dir` and check `requested` against them.
/// Call with the database lock held, before anything is written.
///
/// # Errors
/// `DimensionMismatch` or `SettingsMismatch` on a conflict, `Corrupt` /
/// `ChecksumMismatch` for a damaged settings file, or I/O.
pub(super) fn check(
    dir: &Path,
    dimension: usize,
    requested: &Requested,
) -> Result<Option<IndexSettings>> {
    let Some(settings) = stored::load(dir)? else {
        return Ok(None);
    };
    if settings.dimension != dimension {
        return Err(Error::DimensionMismatch {
            expected: settings.dimension,
            actual: dimension,
        });
    }
    let (kind, ok) = match requested {
        Requested::Unspecified => return Ok(Some(settings)),
        Requested::KindOnly(kind) => (kind, same_kind(&settings.kind, kind)),
        Requested::Exact(kind) => (kind, same_structure(&settings.kind, kind)),
    };
    if !ok {
        return Err(Error::SettingsMismatch(format!(
            "{} was created in {} mode ({:?}) but was opened as {} ({:?}); nothing was \
             changed. Open it with SrvDB::open_existing() or with the recorded settings",
            dir.display(),
            settings.kind.name(),
            settings.kind,
            kind.name(),
            kind
        )));
    }
    Ok(Some(settings))
}

impl SrvDB {
    /// The settings that describe this database as it is now
    pub(crate) fn current_settings(&self) -> IndexSettings {
        let scalar = self.scalar_storage.is_some()
            || (self.config.enabled && self.config.mode == types::QuantizationMode::Scalar);
        let product = self.quantized_storage.is_some() || (self.config.enabled && !scalar);
        let graph = self.hnsw_index.as_ref().map(|h| hnsw_params(h.config()));
        let kind = if let Some(config) = self.rabitq_config {
            IndexKind::Rabitq(config)
        } else if self.current_mode == IndexMode::Auto {
            IndexKind::Auto
        } else if scalar {
            IndexKind::Sq8
        } else if product {
            graph.map_or(IndexKind::Pq, IndexKind::HnswPq)
        } else if let Some(p) = graph {
            // Search uses the graph before IVF partitions
            IndexKind::Hnsw(p)
        } else if let Some(ref ivf) = self.ivf_index {
            IndexKind::Ivf {
                nlist: ivf.config.nlist,
                nprobe: ivf.config.nprobe,
                hnsw: hnsw_params(&ivf.config.hnsw_config),
            }
        } else {
            IndexKind::Flat
        };
        IndexSettings {
            dimension: self.dimension,
            kind,
        }
    }

    /// Write `index_settings.bin` if it differs from the database's
    /// current settings. Atomic: a crash leaves the old or the new file.
    pub(super) fn record_settings(&mut self) -> Result<()> {
        let current = self.current_settings();
        if self.stored_settings != Some(current) {
            stored::save(&self.path, &current)?;
            log::info!("recorded {} index settings", current.kind.name());
            self.stored_settings = Some(current);
        }
        Ok(())
    }

    /// End of every constructor: remember the stored settings, or record
    /// them now when the database is new or was opened with an explicit
    /// mode (the 0.3 / early 0.4 migration path).
    pub(super) fn finish_settings(
        &mut self,
        stored: Option<IndexSettings>,
        record_now: bool,
    ) -> Result<()> {
        match stored {
            Some(settings) => self.stored_settings = Some(settings),
            None if record_now => self.record_settings()?,
            None => {}
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
