//! `Database` class: construction and id-map bookkeeping

use crate::errors::to_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[pyclass(name = "Database", module = "srvdb")]
pub struct PyDatabase {
    pub(crate) db: srvdb::SrvDB,
    pub(crate) id_map: FxHashMap<String, u64>,
    pub(crate) reverse_id_map: FxHashMap<u64, String>,
    pub(crate) dimension: usize,
    pub(crate) index_type: String,
}

impl PyDatabase {
    /// Wrap a database, rebuilding the string-id maps from the `__id__`
    /// field persisted in each metadata entry.
    fn from_db(db: srvdb::SrvDB, dimension: usize, index_type: String) -> PyResult<Self> {
        let mut this = Self {
            db,
            id_map: FxHashMap::default(),
            reverse_id_map: FxHashMap::default(),
            dimension,
            index_type,
        };
        this.load_id_maps()?;
        Ok(this)
    }

    /// Rebuild the string-id maps from metadata (on open and after
    /// compaction, which renumbers internal ids).
    pub(crate) fn load_id_maps(&mut self) -> PyResult<()> {
        self.id_map.clear();
        self.reverse_id_map.clear();
        for (internal_id, meta) in self.db.all_metadata().map_err(to_py)? {
            // Entries written through the Rust API need not be JSON objects
            // with a string id; they are simply not addressable by string id.
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&meta) else {
                continue;
            };
            if let Some(id) = value.get("__id__").and_then(|v| v.as_str()) {
                self.id_map.insert(id.to_string(), internal_id);
                self.reverse_id_map.insert(internal_id, id.to_string());
            }
        }
        Ok(())
    }
}

#[pymethods]
impl PyDatabase {
    /// Create new database with specified dimension (128-4096)
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension (default: 1536)
    ///     mode: Index mode - 'flat', 'hnsw', 'sq8', 'pq' (default: 'flat')
    ///
    /// Examples:
    ///     >>> db = srvdb.Database("./db", dimension=384)  # MiniLM
    ///     >>> db = srvdb.Database("./db", dimension=768, mode='sq8')  # Cohere
    #[new]
    #[pyo3(signature = (path, dimension=1536, mode="flat"))]
    fn new(path: String, dimension: usize, mode: &str) -> PyResult<Self> {
        // Validate dimension
        if !(128..=4096).contains(&dimension) {
            return Err(PyValueError::new_err(format!(
                "Dimension must be between 128 and 4096, got {}. \
                Common values: 384 (MiniLM), 768 (MPNet/Cohere), 1024 (Cohere v3), 1536 (OpenAI)",
                dimension
            )));
        }

        let mut config = srvdb::types::DatabaseConfig::new(dimension)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Set mode
        match mode.to_lowercase().as_str() {
            "flat" => {
                config.index_type = srvdb::types::IndexType::Flat;
            }
            "hnsw" => {
                config.index_type = srvdb::types::IndexType::HNSW;
            }
            "sq8" | "scalar" => {
                config.index_type = srvdb::types::IndexType::ScalarQuantized;
                config.quantization.enabled = true;
                config.quantization.mode = srvdb::types::QuantizationMode::Scalar;
            }
            "auto" => {
                // Auto mode logic will be triggered after init
                // For init config, we start safer (Flat) or let Apply Strategy decide later.
                // We'll mark it as Flat initially but return the object with IndexMode::Auto
                config.index_type = srvdb::types::IndexType::Flat;
            }
            "pq" | "product" => {
                config.index_type = srvdb::types::IndexType::ProductQuantized;
                config.quantization.enabled = true;
                config.quantization.mode = srvdb::types::QuantizationMode::Product;
                config.auto_tune_pq(); // Auto-calculate M and d_sub
            }
            "ivf" => {
                config.index_type = srvdb::types::IndexType::IVF;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid mode '{}'. Choose: 'flat', 'hnsw', 'sq8', 'pq', or 'auto'",
                    mode
                )));
            }
        }

        let mut db = srvdb::SrvDB::new_with_config(&path, config).map_err(to_py)?;

        // Apply Auto Strategy if requested
        if mode.to_lowercase() == "auto" {
            db.set_mode(srvdb::IndexMode::Auto);
        } else if mode.to_lowercase() == "sq8" || mode.to_lowercase() == "scalar" {
            db.current_mode = srvdb::IndexMode::Sq8;
        } else if mode.to_lowercase() == "hnsw" {
            db.current_mode = srvdb::IndexMode::Hnsw;
        } else if mode.to_lowercase() == "ivf" {
            // We need to set mode explicitly to trigger IVF init
            db.set_mode(srvdb::IndexMode::Ivf);
        }

        Self::from_db(db, dimension, mode.to_string())
    }

    /// Create database with HNSW indexing
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension
    ///     m: Graph connectivity (default: 16)
    ///     ef_construction: Build quality (default: 200)
    ///     ef_search: Search quality (default: 50)
    #[staticmethod]
    #[pyo3(signature = (path, dimension=1536, m=16, ef_construction=200, ef_search=50))]
    fn new_hnsw(
        path: String,
        dimension: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> PyResult<Self> {
        let _config = srvdb::types::DatabaseConfig::new(dimension)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // config.index_type = srvdb::types::IndexType::HNSW;

        let hnsw_config = srvdb::hnsw::HNSWConfig::new(m, ef_construction, ef_search);
        let db = srvdb::SrvDB::new_with_hnsw(&path, dimension, hnsw_config).map_err(to_py)?;

        Self::from_db(db, dimension, "hnsw".to_string())
    }

    /// Create database with Scalar Quantization (4x compression)
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension
    ///     training_vectors: Sample vectors for training (1000+ recommended)
    ///
    /// Notes:
    ///     - 4x compression (float32 -> uint8)
    ///     - 95%+ recall typical
    ///     - Much faster than PQ
    #[staticmethod]
    fn new_scalar_quantized(
        path: String,
        dimension: usize,
        training_vectors: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        if training_vectors.is_empty() {
            return Err(PyValueError::new_err(
                "Training vectors required. Provide 1000+ sample vectors for best results.",
            ));
        }

        // Validate dimensions
        for (i, vec) in training_vectors.iter().enumerate() {
            if vec.len() != dimension {
                return Err(PyValueError::new_err(format!(
                    "Training vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    dimension
                )));
            }
        }

        let db = srvdb::SrvDB::new_scalar_quantized(&path, dimension, &training_vectors)
            .map_err(to_py)?;

        Self::from_db(db, dimension, "sq8".to_string())
    }

    /// Create database with Product Quantization (32x compression)
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension
    ///     training_vectors: Sample vectors for training (1000+ recommended)
    ///
    /// Notes:
    ///     - 32x compression (6KB -> 192 bytes)
    ///     - Good recall with re-ranking
    ///     - Supports any dimension divisible by M
    #[staticmethod]
    fn new_product_quantized(
        path: String,
        dimension: usize,
        training_vectors: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        if training_vectors.is_empty() {
            return Err(PyValueError::new_err(
                "Training vectors required. Provide 1000+ sample vectors for best results.",
            ));
        }

        // Validate dimensions
        for (i, vec) in training_vectors.iter().enumerate() {
            if vec.len() != dimension {
                return Err(PyValueError::new_err(format!(
                    "Training vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    dimension
                )));
            }
        }

        // Convert to internal Vector type
        let internal_vectors: Vec<srvdb::Vector> = training_vectors
            .into_iter()
            .map(srvdb::Vector::new)
            .collect();

        let db = srvdb::SrvDB::new_quantized(&path, &internal_vectors).map_err(to_py)?;

        Self::from_db(db, dimension, "pq".to_string())
    }
}
