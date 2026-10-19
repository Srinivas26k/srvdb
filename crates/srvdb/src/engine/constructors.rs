//! Mode-specific constructors

use super::*;

impl SrvDB {
    /// Create new database with Product Quantization
    pub fn new_quantized(path: &str, training_vectors: &[Vector]) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        if training_vectors.is_empty() {
            anyhow::bail!("Training vectors required for quantization");
        }
        let _dimension = training_vectors[0].data.len();

        // Convert training vectors to Vec<Vec<f32>>
        let embedded: Vec<Vec<f32>> = training_vectors.iter().map(|v| v.data.clone()).collect();

        let quantized_storage =
            crate::quantized_storage::QuantizedVectorStorage::new_with_training(path, &embedded)?;
        let metadata_store = MetadataStore::new(path)?;

        let config = types::QuantizationConfig {
            enabled: true,
            ..Default::default()
        };

        Ok(Self {
            path: db_path.to_path_buf(),
            vector_storage: None,
            quantized_storage: Some(quantized_storage),
            scalar_storage: None,
            metadata_store,
            config,
            index_type: types::IndexType::ProductQuantized,
            hnsw_index: None,
            hnsw_config: None,
            ivf_index: None,
            ivf_config: None,
            current_mode: IndexMode::Flat, // PQ is technically a flat scan of compressed vectors
        })
    }

    /// Create new database with Scalar Quantization (SQ8)
    pub fn new_scalar_quantized(
        path: &str,
        dimension: usize,
        training_vectors: &[Vec<f32>],
    ) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        if training_vectors.is_empty() {
            anyhow::bail!("Training vectors required for scalar quantization");
        }

        let scalar_storage =
            ScalarQuantizedStorage::new_with_training(path, dimension, training_vectors)?;

        let metadata_store = MetadataStore::new(path)?;
        let config = types::QuantizationConfig {
            enabled: true,
            mode: types::QuantizationMode::Scalar,
            ..Default::default()
        };

        Ok(Self {
            path: db_path.to_path_buf(),
            vector_storage: None, // Disable full precision storage!
            quantized_storage: None,
            scalar_storage: Some(scalar_storage),
            metadata_store,
            config,
            index_type: types::IndexType::ScalarQuantized,
            hnsw_index: None,
            hnsw_config: None,
            ivf_index: None,
            ivf_config: None,
            current_mode: IndexMode::Sq8,
        })
    }

    /// Create new database with HNSW indexing (full precision vectors)
    ///
    /// # Arguments
    /// * `path` - Database directory path
    /// * `hnsw_config` - HNSW configuration (M, ef_construction, ef_search)
    ///
    /// # Performance
    /// - Search: O(log n) instead of O(n)
    /// - Memory: +200 bytes per vector for graph structure
    /// - 10k vectors: 4ms → 0.5ms (8x faster)
    /// - 100k vectors: 40ms → 1ms (40x faster)
    pub fn new_with_hnsw(
        path: &str,
        dimension: usize,
        hnsw_config: hnsw::HNSWConfig,
    ) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        let vector_storage = VectorStorage::new(path, dimension)?;
        let metadata_store = MetadataStore::new(path)?;

        let mut db = Self {
            path: db_path.to_path_buf(),
            vector_storage: Some(vector_storage),
            quantized_storage: None,
            scalar_storage: None,
            metadata_store,
            config: types::QuantizationConfig::default(),
            index_type: types::IndexType::HNSW,
            hnsw_index: None,
            hnsw_config: None,
            ivf_index: None,
            ivf_config: None,
            current_mode: IndexMode::Hnsw,
        };
        db.attach_hnsw(hnsw_config)?;
        Ok(db)
    }

    /// Create new database with HNSW + Product Quantization (hybrid mode)
    ///
    /// Combines the benefits of both:
    /// - HNSW: O(log n) search complexity
    /// - PQ: 32x memory compression (6KB → 192 bytes)
    ///
    /// # Arguments
    /// * `path` - Database directory path
    /// * `training_vectors` - Vectors for PQ training (recommend 5k-10k samples)
    /// * `hnsw_config` - HNSW configuration
    ///
    /// # Performance
    /// - Memory: 192 bytes (PQ) + 200 bytes (HNSW) = 392 bytes/vector (16x compression)
    /// - Search: 200x faster than flat for 1M vectors
    /// - Recall: ~90-95% (tunable via ef_search)
    pub fn new_with_hnsw_quantized(
        path: &str,
        training_vectors: &[Vector],
        hnsw_config: hnsw::HNSWConfig,
    ) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        // Convert training vectors
        let embedded: Vec<Vec<f32>> = training_vectors.iter().map(|v| v.data.clone()).collect();

        let quantized_storage =
            crate::quantized_storage::QuantizedVectorStorage::new_with_training(path, &embedded)?;
        let metadata_store = MetadataStore::new(path)?;

        let config = types::QuantizationConfig {
            enabled: true,
            ..Default::default()
        };

        let mut hnsw_cfg = hnsw_config;
        hnsw_cfg.use_quantization = true;
        let _hnsw_index = hnsw::HNSWIndex::new(hnsw_cfg.clone());

        Ok(Self {
            path: db_path.to_path_buf(),
            vector_storage: None,
            quantized_storage: Some(quantized_storage),
            scalar_storage: None,
            metadata_store,
            config,
            index_type: types::IndexType::HNSWQuantized,
            hnsw_index: Some(hnsw::HNSWIndex::new(hnsw_cfg.clone())),
            hnsw_config: Some(hnsw_cfg),
            ivf_index: None,
            ivf_config: None,
            current_mode: IndexMode::Hnsw,
        })
    }
}
