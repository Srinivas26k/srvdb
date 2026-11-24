//! `Database` constructors with explicit index parameters: HNSW, SQ8, PQ

use crate::database::PyDatabase;
use crate::errors::to_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Reject an empty training sample or one whose vectors have the wrong
/// dimension
fn check_training(dimension: usize, training_vectors: &[Vec<f32>]) -> PyResult<()> {
    if training_vectors.is_empty() {
        return Err(PyValueError::new_err(
            "Training vectors required. Provide 1000+ sample vectors for best results.",
        ));
    }
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
    Ok(())
}

#[pymethods]
impl PyDatabase {
    /// Open or create an HNSW database with explicit graph parameters.
    ///
    /// Args:
    ///     path: Directory holding the database files; used as given.
    ///     dimension: Vector dimension, 128 to 4096 (default 1536).
    ///     m: Neighbours per graph node (default 16).
    ///     ef_construction: Candidate list size while building (default 200).
    ///     ef_search: Candidate list size while searching (default 50);
    ///         change it later with set_ef_search().
    ///
    /// Returns:
    ///     The open Database, in "hnsw" mode.
    ///
    /// Raises:
    ///     ValueError: dimension outside 128-4096.
    ///     BlockingIOError: the database is already open elsewhere.
    ///     OSError: the files cannot be created or read.
    ///     RuntimeError: stored data is damaged or an internal fault occurred.
    #[staticmethod]
    #[pyo3(signature = (path, dimension=1536, m=16, ef_construction=200, ef_search=50))]
    fn new_hnsw(
        path: String,
        dimension: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> PyResult<Self> {
        srvdb::types::DatabaseConfig::new(dimension)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let hnsw_config = srvdb::hnsw::HNSWConfig::new(m, ef_construction, ef_search);
        let db = srvdb::SrvDB::new_with_hnsw(&path, dimension, hnsw_config).map_err(to_py)?;
        Self::from_db(db, dimension, "hnsw".to_string())
    }

    /// Open or create a scalar-quantized (SQ8) database: one byte per value
    /// instead of four.
    ///
    /// Args:
    ///     path: Directory holding the database files; used as given.
    ///     dimension: Vector dimension; every training vector must have it.
    ///     training_vectors: Sample vectors the quantizer is trained on
    ///         (1000+ recommended). They are not added to the database.
    ///
    /// Returns:
    ///     The open Database, in "sq8" mode.
    ///
    /// Raises:
    ///     ValueError: training_vectors is empty or a vector has the wrong
    ///         dimension.
    ///     BlockingIOError: the database is already open elsewhere.
    ///     OSError: the files cannot be created or read.
    ///     RuntimeError: stored data is damaged or an internal fault occurred.
    #[staticmethod]
    fn new_scalar_quantized(
        path: String,
        dimension: usize,
        training_vectors: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        check_training(dimension, &training_vectors)?;
        let db = srvdb::SrvDB::new_scalar_quantized(&path, dimension, &training_vectors)
            .map_err(to_py)?;
        Self::from_db(db, dimension, "sq8".to_string())
    }

    /// Open or create a product-quantized (PQ) database: vectors are stored
    /// as short codes, and scores are approximate.
    ///
    /// Args:
    ///     path: Directory holding the database files; used as given.
    ///     dimension: Vector dimension; every training vector must have it.
    ///     training_vectors: Sample vectors the codebooks are trained on
    ///         (1000+ recommended). They are not added to the database.
    ///
    /// Returns:
    ///     The open Database, in "pq" mode.
    ///
    /// Raises:
    ///     ValueError: training_vectors is empty or a vector has the wrong
    ///         dimension.
    ///     BlockingIOError: the database is already open elsewhere.
    ///     OSError: the files cannot be created or read.
    ///     RuntimeError: stored data is damaged or an internal fault occurred.
    #[staticmethod]
    fn new_product_quantized(
        path: String,
        dimension: usize,
        training_vectors: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        check_training(dimension, &training_vectors)?;
        let internal_vectors: Vec<srvdb::Vector> = training_vectors
            .into_iter()
            .map(srvdb::Vector::new)
            .collect();
        let db = srvdb::SrvDB::new_quantized(&path, &internal_vectors).map_err(to_py)?;
        Self::from_db(db, dimension, "pq".to_string())
    }
}
