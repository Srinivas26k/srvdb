//! `Database` RaBitQ constructor and settings (ADR 0008)

use crate::database::PyDatabase;
use crate::errors::{non_negative, to_py};
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
    fn set_rabitq_rerank(&self, py: Python<'_>, rerank: i64) -> PyResult<()> {
        let rerank = non_negative("rerank", rerank)?;
        self.write(py, |s| s.db.set_rabitq_rerank(rerank).map_err(to_py))
    }
}

/// Results of a certified or deadline search (ADR 0017): the usual
/// `(id, score)` tuples plus how the search went
#[pyclass(name = "SearchOutcome", module = "srvdb._low_level", frozen)]
pub(crate) struct PySearchOutcome {
    results: Vec<(String, f32)>,
    /// Fraction of live vectors scanned, 0.0 to 1.0
    #[pyo3(get)]
    completeness: f64,
    /// Certified search: lower bound on the probability that the results
    /// are exactly the top k of the scanned vectors; None otherwise
    #[pyo3(get)]
    confidence: Option<f64>,
    /// Results certified without rescoring
    #[pyo3(get)]
    certified: usize,
    /// Candidates rescored on full vectors
    #[pyo3(get)]
    reranked: usize,
    /// Live vectors scanned
    #[pyo3(get)]
    scanned: usize,
}

#[pymethods]
impl PySearchOutcome {
    /// The (id, score) tuples, best first
    #[getter]
    fn results(&self) -> Vec<(String, f32)> {
        self.results.clone()
    }

    fn __len__(&self) -> usize {
        self.results.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<(String, f32)> {
        let n = self.results.len() as isize;
        let i = if index < 0 { index + n } else { index };
        if i < 0 || i >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "SearchOutcome index out of range",
            ));
        }
        Ok(self.results[i as usize].clone())
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = pyo3::types::PyList::new(py, self.results.clone())?;
        Ok(list.try_iter()?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchOutcome({} results, completeness={}, confidence={:?})",
            self.results.len(),
            self.completeness,
            self.confidence
        )
    }
}

impl PyDatabase {
    /// `search(..., certify=, deadline_ms=)`: argument conversion only
    pub(crate) fn search_rabitq(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        k: i64,
        certify: bool,
        deadline_ms: Option<f64>,
    ) -> PyResult<PySearchOutcome> {
        let k = non_negative("k", k)?;
        let mut options = srvdb::RabitqSearch::new().with_certify(certify);
        if let Some(ms) = deadline_ms {
            if !(ms.is_finite() && ms >= 0.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "deadline_ms must be a finite number >= 0, got {ms}"
                )));
            }
            let deadline = std::time::Duration::try_from_secs_f64(ms / 1000.0)
                .unwrap_or(std::time::Duration::MAX);
            options = options.with_deadline(deadline);
        }
        let query = srvdb::Vector::new(query);
        self.read(py, |s| {
            let out =
                s.db.search_rabitq_with(&query, k, &options)
                    .map_err(to_py)?;
            Ok(PySearchOutcome {
                results: crate::operations::named(s, out.results),
                completeness: out.stats.completeness,
                confidence: out.stats.confidence,
                certified: out.stats.certified,
                reranked: out.stats.reranked,
                scanned: out.stats.scanned,
            })
        })
    }
}
