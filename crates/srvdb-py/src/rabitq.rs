//! `Database` RaBitQ constructor and settings (ADR 0008)

use crate::database::PyDatabase;
use crate::errors::to_py;
use pyo3::prelude::*;

/// Bit width named by a mode string ('rabitq1', 'rabitq2', 'rabitq4')
pub(crate) fn bits_for_mode(mode: &str) -> Option<u8> {
    match mode.to_lowercase().as_str() {
        "rabitq1" => Some(1),
        "rabitq2" => Some(2),
        "rabitq4" => Some(4),
        _ => None,
    }
}

impl PyDatabase {
    pub(crate) fn open_rabitq(
        path: String,
        dimension: usize,
        bits: u8,
        rerank: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut config = srvdb::RabitqConfig::new(bits).map_err(to_py)?;
        if let Some(rerank) = rerank {
            config = config.with_rerank(rerank);
        }
        if let Some(seed) = seed {
            config = config.with_seed(seed);
        }
        let db = srvdb::SrvDB::new_rabitq(&path, dimension, config).map_err(to_py)?;
        Self::from_db(db, dimension, format!("rabitq{bits}"))
    }
}

#[pymethods]
impl PyDatabase {
    /// Create or open a database searched through RaBitQ codes
    ///
    /// Full vectors are kept on disk; searches scan 1, 2 or 4-bit codes
    /// for k * rerank candidates and rescore them exactly. No training.
    ///
    /// Args:
    ///     path: Database directory
    ///     dimension: Vector dimension
    ///     bits: Bits per dimension, 1, 2 or 4 (default: 4)
    ///     rerank: Candidates per result to rescore; 0 disables reranking
    ///         (default: 40, 20 or 10 for 1, 2 or 4 bits)
    ///     seed: Rotation seed; changing it rebuilds the codes on open
    ///
    /// Raises:
    ///     ValueError: unsupported bits, rerank too large, bad dimension
    #[staticmethod]
    #[pyo3(signature = (path, dimension=1536, bits=4, rerank=None, seed=None))]
    fn new_rabitq(
        path: String,
        dimension: usize,
        bits: u8,
        rerank: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Self::open_rabitq(path, dimension, bits, rerank, seed)
    }

    /// Set the RaBitQ rerank factor for later searches (0 = no rerank)
    ///
    /// Raises:
    ///     ValueError: not a RaBitQ database, or the factor is too large
    fn set_rabitq_rerank(&self, py: Python<'_>, rerank: usize) -> PyResult<()> {
        self.write(py, |s| s.db.set_rabitq_rerank(rerank).map_err(to_py))
    }
}
