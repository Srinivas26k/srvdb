//! Opening a database with the settings recorded on disk (ADR 0011)

use super::settings::{hnsw_config, Requested};
use super::*;
use crate::storage::settings::{self as stored, IndexKind};
use types::{IndexType, QuantizationMode};

/// The configuration to reopen with; the dimension is checked against the
/// header limit shared with `DatabaseConfig` (`index::limits`).
///
/// Not `DatabaseConfig::new` / `validate`: the 128 floor would lock out
/// databases the core creates with smaller dimensions (`new_with_hnsw`,
/// `new_scalar_quantized`, ...), and `validate` checks PQ `m` = 192 from
/// the default config (for SQ8 too), while a stored PQ database uses the
/// `m` of its checksummed codebooks.
pub(super) fn config_for(
    dimension: usize,
    quant: Option<QuantizationMode>,
    index_type: IndexType,
) -> Result<types::DatabaseConfig> {
    crate::index::limits::check(
        "dimension",
        dimension,
        1,
        crate::index::limits::MAX_DIMENSION,
    )?;
    let mut config = types::DatabaseConfig {
        dimension,
        quantization: types::QuantizationConfig::default(),
        index_type,
    };
    if let Some(mode) = quant {
        config.quantization.enabled = true;
        config.quantization.mode = mode;
    }
    Ok(config)
}

impl SrvDB {
    /// Open an existing database in the mode it was created with, reading
    /// the dimension, index mode and its parameters from
    /// `index_settings.bin`. Nothing has to be passed but the path.
    ///
    /// A database written by SrvDB 0.3 or an earlier 0.4 build has no
    /// recorded settings: if it holds full-precision vectors it opens flat
    /// as before (an existing `hnsw.graph` or `ivf.index` is still used);
    /// other modes must be opened once with their constructor, which
    /// records the settings.
    ///
    /// Durability and thread-safety are as for every constructor (ADR 0003,
    /// ADR 0005).
    ///
    /// # Errors
    /// `NotFound` if `path` holds no SrvDB database, `InvalidState` for
    /// a quantized database without recorded settings, `Corrupt` /
    /// `ChecksumMismatch` for a damaged settings or data file (including an
    /// unknown index kind), `Locked`, or I/O.
    pub fn open_existing(path: &str) -> Result<Self> {
        let dir = Path::new(path);
        if !dir.is_dir() {
            return Err(Error::NotFound(format!(
                "{} is not a SrvDB database directory; create the database first",
                dir.display()
            )));
        }
        let Some(settings) = stored::load(dir)? else {
            return Self::open_legacy(path);
        };
        let dim = settings.dimension;
        let open = |quant, index_type| {
            Self::open_config(
                path,
                config_for(dim, quant, index_type)?,
                Requested::Unspecified,
            )
        };
        let mut db = match settings.kind {
            IndexKind::Hnsw(p) => Self::new_with_hnsw(path, dim, hnsw_config(&p))?,
            IndexKind::Rabitq(c) => Self::new_rabitq(path, dim, c)?,
            IndexKind::Sq8 => open(Some(QuantizationMode::Scalar), IndexType::ScalarQuantized)?,
            IndexKind::Pq | IndexKind::HnswPq(_) => {
                open(Some(QuantizationMode::Product), IndexType::ProductQuantized)?
            }
            IndexKind::Flat | IndexKind::Ivf { .. } | IndexKind::Auto => {
                open(None, IndexType::Flat)?
            }
        };
        if settings.kind == IndexKind::Auto {
            db.set_mode(IndexMode::Auto);
        }
        Ok(db)
    }

    /// A database without `index_settings.bin`: flat, as SrvDB 0.3 opened it
    fn open_legacy(path: &str) -> Result<Self> {
        let dir = Path::new(path);
        let file = dir.join(storage::flat::FILE_NAME);
        if !file.exists() {
            let quantized = [storage::sq::FILE_NAME, storage::pq::FILE_NAME]
                .iter()
                .any(|name| dir.join(name).exists());
            return Err(if quantized {
                Error::invalid_state(format!(
                    "{} was written by an older SrvDB without recorded index settings \
                     and is quantized; open it once with its mode (for example \
                     new_scalar_quantized or new_with_config) to record them",
                    dir.display()
                ))
            } else {
                Error::NotFound(format!(
                    "{} holds no SrvDB database; create it first",
                    dir.display()
                ))
            });
        }
        let mut header = [0u8; types::VectorHeader::SIZE];
        {
            use std::io::Read;
            let mut f = std::fs::File::open(&file)?;
            f.read_exact(&mut header).map_err(|e| {
                Error::Corrupt(format!(
                    "{} is shorter than its header ({e}); restore it from a backup",
                    file.display()
                ))
            })?;
        }
        let dimension = types::VectorHeader::from_bytes(&header)?.dimension as usize;
        let config = config_for(dimension, None, IndexType::Flat)?;
        Self::open_config(path, config, Requested::Unspecified)
    }
}
