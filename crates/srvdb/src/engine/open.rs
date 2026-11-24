//! Opening a database from a configuration

use super::*;

impl SrvDB {
    /// Create or open a database with configuration.
    ///
    /// Quantized modes reopen existing quantizer files; for a new database
    /// the quantizer is trained on the first batch of vectors added.
    ///
    /// Index settings (ADR 0011): a default configuration (flat, no
    /// quantization) opens an existing database in the mode recorded in
    /// `index_settings.bin`, like [`SrvDB::open_existing`]. A configuration
    /// naming a mode (an index type or quantization) must match the
    /// recorded mode. A new database, or one without recorded settings that
    /// is opened with a mode, has its settings recorded before this returns.
    ///
    /// # Errors
    /// `SettingsMismatch` if the requested mode differs from the recorded
    /// one, `DimensionMismatch` for another dimension, `Locked`, `Corrupt` /
    /// `ChecksumMismatch` for damaged files, or I/O.
    pub fn new_with_config(path: &str, config: types::DatabaseConfig) -> Result<Self> {
        let requested = requested_by(&config);
        if matches!(requested, settings::Requested::Unspecified) {
            // Read without the lock only to pick the constructor; the one
            // that runs checks the settings again under the lock
            if let Some(stored) = storage::settings::load(Path::new(path))? {
                if stored.kind != storage::settings::IndexKind::Flat {
                    if stored.dimension != config.dimension {
                        return Err(Error::DimensionMismatch {
                            expected: stored.dimension,
                            actual: config.dimension,
                        });
                    }
                    return Self::open_existing(path);
                }
            }
        }
        Self::open_config(path, config, requested)
    }

    /// `new_with_config` without the redirect to the stored mode
    pub(super) fn open_config(
        path: &str,
        config: types::DatabaseConfig,
        requested: settings::Requested,
    ) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        let dimension = config.dimension;
        let lock = storage::lock::acquire(Path::new(path))?;
        let fresh = settings::is_fresh(db_path);
        let stored = settings::check(db_path, dimension, &requested)?;
        let metadata_store = MetadataStore::new(path)?;

        let quant = &config.quantization;
        let (mut vector_storage, mut quantized_storage, mut scalar_storage) = (None, None, None);
        let mut current_mode = IndexMode::Flat;
        if quant.enabled && quant.mode == types::QuantizationMode::Scalar {
            // Trained means the quantizer file exists: a crash during the
            // first training can leave sq8_vectors.bin without it, and such
            // a database simply trains again on the next batch.
            if db_path.join(storage::sq::QUANTIZER_FILE).exists() {
                scalar_storage = Some(ScalarQuantizedStorage::load(path, dimension)?);
            }
            current_mode = IndexMode::Sq8;
        } else if quant.enabled && quant.mode == types::QuantizationMode::Product {
            if db_path.join(storage::pq::CODEBOOK_FILE).exists() {
                quantized_storage = Some(QuantizedVectorStorage::load(path)?);
            }
        } else {
            vector_storage = Some(VectorStorage::new(path, dimension)?);
        }

        let mut db = Self {
            path: db_path.to_path_buf(),
            vector_storage,
            quantized_storage,
            scalar_storage,
            metadata_store,
            config: config.quantization,
            index_type: config.index_type,
            hnsw_index: None,
            hnsw_config: None,
            ivf_index: None,
            ivf_config: None,
            rabitq: None,
            rabitq_config: None,
            stored_settings: None,
            current_mode,
            dimension,
            deleted: Default::default(),
            pending_deletes: Default::default(),
            _lock: lock,
        };
        db.recover()?;

        use storage::settings::IndexKind;
        let stored_kind = stored.map(|s| s.kind);
        let stored_graph = match stored_kind {
            Some(IndexKind::Hnsw(p)) | Some(IndexKind::HnswPq(p)) => Some(p),
            _ => None,
        };
        if config.index_type == types::IndexType::HNSW
            || stored_graph.is_some()
            || db_path.join("hnsw.graph").exists()
        {
            let mut graph =
                stored_graph.map_or_else(Default::default, |p| settings::hnsw_config(&p));
            graph.use_quantization = db.quantized_storage.is_some()
                || (db.config.enabled && db.config.mode == types::QuantizationMode::Product);
            db.attach_hnsw(graph)?;
        }
        if let Some(ivf) = storage::ivf::IVFStorage::load(db_path)? {
            db.attach_ivf(ivf)?;
        } else if let Some(IndexKind::Ivf {
            nlist,
            nprobe,
            hnsw,
        }) = stored_kind
        {
            db.ivf_config = Some(ivf::IVFConfig {
                nlist,
                nprobe,
                hnsw_config: settings::hnsw_config(&hnsw),
                ..Default::default()
            });
            db.set_mode(IndexMode::Ivf);
        } else if config.index_type == types::IndexType::IVF && db.ivf_index.is_none() {
            db.set_mode(IndexMode::Ivf);
        }

        let explicit = !matches!(requested, settings::Requested::Unspecified);
        db.finish_settings(stored, fresh || explicit)?;
        Ok(db)
    }

    /// Enable HNSW: load `hnsw.graph` if present (its stored config wins),
    /// otherwise start a new graph. Stored vectors missing from the graph
    /// (e.g. added after the last persist) are indexed.
    pub(crate) fn attach_hnsw(&mut self, config: hnsw::HNSWConfig) -> Result<()> {
        let graph_path = self.path.join("hnsw.graph");
        let loaded = if graph_path.exists() {
            match std::fs::read(&graph_path)
                .map_err(anyhow::Error::from)
                .and_then(|bytes| hnsw::HNSWIndex::from_bytes(&bytes))
            {
                Ok(index) => Some(index),
                Err(e) => {
                    log::warn!("hnsw.graph could not be loaded, rebuilding the graph: {e}");
                    None
                }
            }
        } else {
            None
        };

        let index = loaded.unwrap_or_else(|| hnsw::HNSWIndex::new(config.clone()));
        let indexed = index.len() as u64;
        self.hnsw_index = Some(index);
        self.hnsw_config = Some(config);
        self.current_mode = IndexMode::Hnsw;

        let count = if let Some(ref qstorage) = self.quantized_storage {
            self.index_type = types::IndexType::HNSWQuantized;
            qstorage.count()
        } else {
            self.index_type = types::IndexType::HNSW;
            self.vector_storage.as_ref().map_or(0, |s| s.count())
        };
        let missing: Vec<u64> = (indexed..count).collect();
        self.index_new_ids(&missing)
    }

    /// Use a saved IVF index: switch to IVF mode and index vectors committed
    /// after the index was saved. An index covering more vectors than are
    /// committed (not produced by SrvDB itself) is retrained instead.
    fn attach_ivf(&mut self, ivf: ivf::IVFIndex) -> Result<()> {
        let covered: u64 = ivf.partitions.iter().map(|p| p.read().len() as u64).sum();
        self.ivf_config = Some(ivf.config.clone());
        self.ivf_index = Some(ivf);
        self.current_mode = IndexMode::Ivf;
        let count = self.count();
        if covered > count {
            log::warn!("ivf.index covers {covered} vectors but {count} are committed; retraining");
            return self.train_ivf();
        }
        let missing: Vec<u64> = (covered..count).collect();
        self.index_into(&missing, false, true)
    }

    /// Train quantized storage from the first vectors added when the
    /// database was opened in a quantized mode without existing data.
    pub(super) fn ensure_quantized_storage(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if !self.config.enabled
            || self.scalar_storage.is_some()
            || self.quantized_storage.is_some()
            || vectors.is_empty()
        {
            return Ok(());
        }

        let path = self
            .path
            .to_str()
            .ok_or_else(|| Error::invalid_input("database path is not valid UTF-8"))?;
        match self.config.mode {
            types::QuantizationMode::Scalar => {
                self.scalar_storage = Some(ScalarQuantizedStorage::new_with_training(
                    path,
                    vectors[0].len(),
                    vectors,
                )?);
            }
            types::QuantizationMode::Product => {
                if vectors.len() < quant::product::K {
                    return Err(Error::InvalidInput(format!(
                        "PQ trains on the first batch added, which needs at least {} vectors \
                        (got {}). Add a larger first batch or use new_product_quantized().",
                        quant::product::K,
                        vectors.len()
                    )));
                }
                self.quantized_storage =
                    Some(QuantizedVectorStorage::new_with_training(path, vectors)?);
            }
            types::QuantizationMode::None => {}
        }
        Ok(())
    }
}

/// The mode a `DatabaseConfig` names, if any
fn requested_by(config: &types::DatabaseConfig) -> settings::Requested {
    use storage::settings::{HnswParams, IndexKind};
    use types::{IndexType, QuantizationMode};
    let q = &config.quantization;
    let kind = if q.enabled && q.mode == QuantizationMode::Scalar {
        IndexKind::Sq8
    } else if q.enabled && q.mode == QuantizationMode::Product {
        IndexKind::Pq
    } else {
        // Parameters are not part of DatabaseConfig: placeholders, only
        // the kind is compared
        let p = HnswParams {
            m: 0,
            ef_construction: 0,
            ef_search: 0,
        };
        match config.index_type {
            IndexType::HNSW => IndexKind::Hnsw(p),
            IndexType::IVF => IndexKind::Ivf {
                nlist: 0,
                nprobe: 0,
                hnsw: p,
            },
            _ => return settings::Requested::Unspecified,
        }
    };
    settings::Requested::KindOnly(kind)
}
