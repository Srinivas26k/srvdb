//! `Database` class: construction and id-map bookkeeping

use crate::errors::to_py;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::RwLock;

/// Everything a Python `Database` owns, behind one lock
pub(crate) struct State {
    pub(crate) db: srvdb::SrvDB,
    pub(crate) id_map: FxHashMap<String, u64>,
    pub(crate) reverse_id_map: FxHashMap<u64, String>,
    pub(crate) dimension: usize,
    pub(crate) index_type: String,
}

impl State {
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

/// A SrvDB database, safe to share between Python threads (ADR 0005).
///
/// Searches share a read lock and run in parallel; writes take the lock
/// exclusively, so other calls wait for them instead of failing. The lock is
/// always waited on with the GIL released: a thread holding the GIL while
/// waiting could block a writer that needs the GIL, and both would hang.
#[pyclass(name = "Database", module = "srvdb", frozen)]
pub struct PyDatabase {
    /// `None` once close() has released the database
    state: RwLock<Option<State>>,
}

/// A panic while holding the lock leaves the state possibly half-updated
fn poisoned() -> PyErr {
    PyRuntimeError::new_err(
        "this Database is unusable after an internal error; open it again \
         (committed data is safe)",
    )
}

/// Use of a database after close(), like I/O on a closed Python file
fn closed() -> PyErr {
    PyValueError::new_err("this Database is closed; open it again with srvdb.open(path)")
}

impl PyDatabase {
    /// Wrap a database, rebuilding the string-id maps from the `__id__`
    /// field persisted in each metadata entry.
    pub(crate) fn from_db(
        db: srvdb::SrvDB,
        dimension: usize,
        index_type: String,
    ) -> PyResult<Self> {
        let mut state = State {
            db,
            id_map: FxHashMap::default(),
            reverse_id_map: FxHashMap::default(),
            dimension,
            index_type,
        };
        state.load_id_maps()?;
        Ok(Self {
            state: RwLock::new(Some(state)),
        })
    }

    /// Run `f` under the shared lock, with the GIL released
    pub(crate) fn read<R: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&State) -> PyResult<R> + Send,
    ) -> PyResult<R> {
        py.detach(|| {
            let guard = self.state.read().map_err(|_| poisoned())?;
            f(guard.as_ref().ok_or_else(closed)?)
        })
    }

    /// Run `f` under the exclusive lock, with the GIL released
    pub(crate) fn write<R: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut State) -> PyResult<R> + Send,
    ) -> PyResult<R> {
        py.detach(|| {
            let mut guard = self.state.write().map_err(|_| poisoned())?;
            f(guard.as_mut().ok_or_else(closed)?)
        })
    }

    /// Drop the database (releasing its lock) and return whether it was
    /// open. Waits for running calls, with the GIL released.
    pub(crate) fn take(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| {
            let mut guard = self.state.write().map_err(|_| poisoned())?;
            Ok(guard.take().is_some())
        })
    }

    /// Whether close() has been called
    pub(crate) fn is_closed(&self, py: Python<'_>) -> bool {
        py.detach(|| self.state.read().is_ok_and(|g| g.is_none()))
    }
}

#[pymethods]
impl PyDatabase {
    /// Open the database at `path`, creating it if it does not exist.
    ///
    /// Kept for code written against SrvDB 0.3; new code uses
    /// `srvdb.create(path, dimension, index)` and `srvdb.open(path)`.
    ///
    /// Args:
    ///     path: Directory holding the database files; used as given.
    ///     dimension: Vector dimension, 128 to 4096 (default 1536). Must
    ///         match the stored dimension when the database exists.
    ///     mode: Index mode, case-insensitive: "flat" (default), "hnsw",
    ///         "sq8" (alias "scalar"), "pq" (alias "product"), "ivf" or
    ///         "auto", or "rabitq1" / "rabitq2" / "rabitq4" (RaBitQ codes with
    ///         rerank, see new_rabitq). The mode is recorded on disk; reopening
    ///         with the default "flat" uses the recorded mode, another mode
    ///         raises ValueError.
    ///
    /// Raises:
    ///     ValueError: dimension outside 128-4096, unknown mode, a mode
    ///         that conflicts with the recorded one, or a
    ///         dimension that does not match the stored database.
    ///     BlockingIOError: the database is already open in this or
    ///         another process.
    ///     OSError: the files cannot be created or read.
    ///     RuntimeError: stored data is damaged (checksum mismatch) or an
    ///         internal fault occurred.
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

        if let Some(bits) = crate::rabitq::bits_for_mode(mode) {
            return Self::open_rabitq(path, dimension, bits, None, None);
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
                    "Invalid mode '{}'. Choose: 'flat', 'hnsw', 'sq8' (or 'scalar'), \
                    'pq' (or 'product'), 'ivf', 'auto', 'rabitq1', 'rabitq2' or 'rabitq4'",
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
}
