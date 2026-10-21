//! Opening a database from a configuration

use super::*;

impl SrvDB {
    /// Create or open a database with configuration.
    ///
    /// Quantized modes reopen existing quantizer files; for a new database
    /// the quantizer is trained on the first batch of vectors added.
    pub fn new_with_config(path: &str, config: types::DatabaseConfig) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        let dimension = config.dimension;
        let metadata_store = MetadataStore::new(path)?;

        let quant = &config.quantization;
        let (mut vector_storage, mut quantized_storage, mut scalar_storage) = (None, None, None);
        let mut current_mode = IndexMode::Flat;
        if quant.enabled && quant.mode == types::QuantizationMode::Scalar {
            if db_path.join("sq8_vectors.bin").exists() {
                scalar_storage = Some(ScalarQuantizedStorage::load(path, dimension)?);
            }
            current_mode = IndexMode::Sq8;
        } else if quant.enabled && quant.mode == types::QuantizationMode::Product {
            if db_path.join("codebooks.bin").exists() {
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
            current_mode,
        };

        if config.index_type == types::IndexType::HNSW || db_path.join("hnsw.graph").exists() {
            db.attach_hnsw(hnsw::HNSWConfig::default())?;
        }

        Ok(db)
    }

    /// Enable HNSW: load `hnsw.graph` if present (its stored config wins),
    /// otherwise start a new graph. Stored vectors missing from the graph
    /// (e.g. added after the last persist) are indexed.
    pub(super) fn attach_hnsw(&mut self, config: hnsw::HNSWConfig) -> Result<()> {
        let graph_path = self.path.join("hnsw.graph");
        let loaded = if graph_path.exists() {
            match std::fs::read(&graph_path)
                .map_err(anyhow::Error::from)
                .and_then(|bytes| hnsw::HNSWIndex::from_bytes(&bytes))
            {
                Ok(index) => Some(index),
                Err(e) => {
                    eprintln!("Failed to load HNSW graph, rebuilding: {}", e);
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
        self.index_type = types::IndexType::HNSW;
        self.current_mode = IndexMode::Hnsw;

        let count = self.vector_storage.as_ref().map_or(0, |s| s.count());
        let missing: Vec<u64> = (indexed..count).collect();
        self.index_new_ids(&missing)
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
            .ok_or_else(|| anyhow::anyhow!("Invalid DB path"))?;
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
                    anyhow::bail!(
                        "PQ trains on the first batch added, which needs at least {} vectors \
                        (got {}). Add a larger first batch or use new_product_quantized().",
                        quant::product::K,
                        vectors.len()
                    );
                }
                self.quantized_storage =
                    Some(QuantizedVectorStorage::new_with_training(path, vectors)?);
            }
            types::QuantizationMode::None => {}
        }
        Ok(())
    }
}
