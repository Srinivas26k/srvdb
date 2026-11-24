//! Creating a new database with fixed settings (ADR 0013)

use super::create_options::{CreateOptions, IndexChoice};
use super::open_existing::config_for;
use super::settings::Requested;
use super::*;
use crate::index::limits;
use crate::quant::rabitq::RabitqConfig;
use types::{IndexType, QuantizationMode};

/// A numeric option as `usize`, or `default` when it was not set
fn usize_option(options: &CreateOptions, name: &str, default: usize) -> Result<usize> {
    match options.get(name) {
        None => Ok(default),
        Some(value) => usize::try_from(value)
            .map_err(|_| Error::invalid_input(format!("{name} is too large, got {value}"))),
    }
}

/// HNSW parameters from the options, defaults from `HNSWConfig::default`
fn hnsw_from(options: &CreateOptions) -> Result<hnsw::HNSWConfig> {
    let d = hnsw::HNSWConfig::default();
    let config = hnsw::HNSWConfig::new(
        usize_option(options, "m", d.m)?,
        usize_option(options, "ef_construction", d.ef_construction)?,
        usize_option(options, "ef_search", d.ef_search)?,
    );
    config.validate()?;
    Ok(config)
}

fn rabitq_from(options: &CreateOptions) -> Result<RabitqConfig> {
    let bits = options.get("bits").unwrap_or(4);
    let bits = u8::try_from(bits)
        .map_err(|_| Error::invalid_input(format!("bits must be 1, 2 or 4, got {bits}")))?;
    let mut config = RabitqConfig::new(bits)?;
    if options.get("rerank").is_some() {
        config = config.with_rerank(usize_option(options, "rerank", 0)?);
    }
    if let Some(seed) = options.get("seed") {
        config = config.with_seed(seed);
    }
    config.validate()?;
    Ok(config)
}

/// The training vectors, each checked against `dimension`
fn training_from(options: &CreateOptions, dimension: usize) -> Result<Option<&[Vec<f32>]>> {
    let Some(vectors) = options.training() else {
        return Ok(None);
    };
    let found = validate::training(vectors.iter().map(Vec::as_slice))?;
    if found != dimension {
        return Err(Error::DimensionMismatch {
            expected: dimension,
            actual: found,
        });
    }
    Ok(Some(vectors))
}

/// Refuse a path that is a file or a directory with anything in it
fn check_new(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    if !dir.is_dir() {
        return Err(Error::AlreadyExists(format!(
            "{} is a file; choose a new or empty directory",
            dir.display()
        )));
    }
    if std::fs::read_dir(dir)?.next().is_none() {
        return Ok(());
    }
    Err(Error::AlreadyExists(if storage::identity::has_magic(dir) {
        format!(
            "a SrvDB database already exists at {}; open it with SrvDB::open_existing \
                 (srvdb.open in Python) or delete it first",
            dir.display()
        )
    } else {
        format!(
            "{} is not empty; choose a new or empty directory",
            dir.display()
        )
    }))
}

impl SrvDB {
    /// Create a new database at `path` with the index and options given.
    ///
    /// The settings are recorded in `index_settings.bin` before this
    /// returns, so [`SrvDB::open_existing`] reopens the database with only
    /// its path. `path` must not exist or be an empty directory; it is
    /// created (with parents) if missing.
    ///
    /// Defaults: HNSW `m` 16, `ef_construction` 200, `ef_search` 50; IVF
    /// `nlist` 100, `nprobe` 10; RaBitQ 4 bits with the default rerank.
    /// SQ8 and PQ train on `training` if given, otherwise on the first
    /// batch added. Durability and thread-safety are as for every
    /// constructor (ADR 0003, ADR 0005). The emptiness check and the
    /// creation are not one atomic step: two processes creating the same
    /// path at once are only kept apart by the database lock.
    ///
    /// With [`CreateOptions::set_exist_ok`], an existing database is opened
    /// instead if its recorded index, dimension and every option given
    /// match; otherwise `SettingsMismatch` / `DimensionMismatch`.
    ///
    /// # Errors
    /// `AlreadyExists` if `path` is a file, a database or a non-empty
    /// directory; `InvalidInput` for a dimension outside 1..=4096 or an
    /// option out of range; `DimensionMismatch` for training vectors of
    /// another dimension; `Locked` or I/O.
    pub fn create(path: &str, dimension: usize, options: &CreateOptions) -> Result<Self> {
        limits::check("dimension", dimension, 1, limits::MAX_DIMENSION)?;
        // Check every option before touching the disk
        let training = training_from(options, dimension)?;
        let index = options.index();
        let hnsw = match index {
            IndexChoice::Hnsw | IndexChoice::Ivf => Some(hnsw_from(options)?),
            _ => None,
        };
        let rabitq = match index {
            IndexChoice::Rabitq => Some(rabitq_from(options)?),
            _ => None,
        };
        let ivf = match index {
            IndexChoice::Ivf => {
                let d = ivf::IVFConfig::default();
                let config = ivf::IVFConfig {
                    nlist: usize_option(options, "nlist", d.nlist)?,
                    nprobe: usize_option(options, "nprobe", d.nprobe)?,
                    hnsw_config: hnsw.clone().unwrap_or_default(),
                    ..d
                };
                config.validate()?;
                Some(config)
            }
            _ => None,
        };
        let dir = Path::new(path);
        if options.exist_ok() && dir.is_dir() && storage::identity::has_magic(dir) {
            return Self::open_matching(path, dimension, options);
        }
        check_new(dir)?;
        log::info!("creating {index} database (dimension {dimension})");

        let open = |quant: Option<QuantizationMode>, index_type: IndexType| {
            let mut config = config_for(dimension, quant, index_type)?;
            if quant == Some(QuantizationMode::Product) {
                config.auto_tune_pq();
            }
            Self::open_config(path, config, Requested::Unspecified)
        };
        let mut db = match (index, hnsw, rabitq) {
            (IndexChoice::Hnsw, Some(config), _) => Self::new_with_hnsw(path, dimension, config)?,
            (IndexChoice::Rabitq, _, Some(config)) => Self::new_rabitq(path, dimension, config)?,
            (IndexChoice::Sq8, _, _) => match training {
                Some(t) => Self::new_scalar_quantized(path, dimension, t)?,
                None => open(Some(QuantizationMode::Scalar), IndexType::ScalarQuantized)?,
            },
            (IndexChoice::Pq, _, _) => match training {
                Some(t) => {
                    let vectors: Vec<Vector> = t.iter().cloned().map(Vector::new).collect();
                    Self::new_quantized(path, &vectors)?
                }
                None => open(Some(QuantizationMode::Product), IndexType::ProductQuantized)?,
            },
            (IndexChoice::Ivf, _, _) => open(None, IndexType::IVF)?,
            (IndexChoice::Auto, _, _) => {
                let mut db = open(None, IndexType::Flat)?;
                db.set_mode(IndexMode::Auto);
                db
            }
            _ => open(None, IndexType::Flat)?,
        };
        if let Some(config) = ivf {
            db.configure_ivf(config)?;
        }
        // Mode changes after construction (IVF parameters, auto) are
        // recorded now, not at the first persist()
        db.record_settings()?;
        Ok(db)
    }

    /// The index mode of this database as recorded in its settings:
    /// "flat", "hnsw", "sq8", "pq", "hnsw+pq", "ivf", "rabitq" or "auto"
    pub fn index_name(&self) -> &'static str {
        self.current_settings().kind.name()
    }

    /// The vector dimension every insert and query must match
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}
