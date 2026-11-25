//! Module verbs `create`, `open`, `delete`, `rebuild_checksums` and the `Database` lifecycle
//! methods `save`, `close`, `__enter__`, `__exit__` (ADR 0013).
//!
//! Option names and values are checked by the core (`srvdb::CreateOptions`);
//! this file only converts the keyword dictionary.

use crate::database::PyDatabase;
use crate::errors::to_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use srvdb::{CreateOptions, IndexChoice, SrvDB, VectorEngine};

fn wrap(db: SrvDB) -> PyResult<PyDatabase> {
    let dimension = db.dimension();
    let name = db.index_name().to_string();
    PyDatabase::from_db(db, dimension, name)
}

/// Convert the keyword options; the core rejects names that do not apply
fn convert(index: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<CreateOptions> {
    let index: IndexChoice = index.parse().map_err(to_py)?;
    let mut converted = CreateOptions::new(index);
    let Some(options) = options else {
        return Ok(converted);
    };
    for (key, value) in options.iter() {
        let name: String = key.extract()?;
        if name == "training" {
            let vectors: Vec<Vec<f32>> = value.extract().map_err(|_| {
                PyValueError::new_err("training must be a sequence of float vectors")
            })?;
            converted.set_training(vectors).map_err(to_py)?;
        } else {
            let number: u64 = value.extract().map_err(|_| {
                PyValueError::new_err(format!("option '{name}' must be a non-negative integer"))
            })?;
            converted.set(&name, number).map_err(to_py)?;
        }
    }
    Ok(converted)
}

/// Create a new database; see `srvdb.create` for the documentation
#[pyfunction]
#[pyo3(signature = (path, dimension, index="hnsw", exist_ok=false, options=None))]
pub(crate) fn create(
    py: Python<'_>,
    path: String,
    dimension: usize,
    index: &str,
    exist_ok: bool,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyDatabase> {
    let mut converted = convert(index, options)?;
    converted.set_exist_ok(exist_ok);
    let db = py
        .detach(|| SrvDB::create(&path, dimension, &converted))
        .map_err(to_py)?;
    wrap(db)
}

/// Open an existing database by path; see `srvdb.open`
#[pyfunction]
pub(crate) fn open(py: Python<'_>, path: String) -> PyResult<PyDatabase> {
    let db = py.detach(|| SrvDB::open_existing(&path)).map_err(to_py)?;
    wrap(db)
}

/// Delete a closed database directory; see `srvdb.delete`
#[pyfunction]
pub(crate) fn delete(py: Python<'_>, path: String) -> PyResult<()> {
    py.detach(|| SrvDB::delete_database(&path)).map_err(to_py)
}

/// Rebuild missing or damaged checksum files; see `srvdb.rebuild_checksums`
#[pyfunction]
pub(crate) fn rebuild_checksums(py: Python<'_>, path: String) -> PyResult<Vec<String>> {
    py.detach(|| SrvDB::rebuild_checksums(&path)).map_err(to_py)
}

#[pymethods]
impl PyDatabase {
    /// Make every earlier write durable (crash-safe commit). ``persist()``
    /// is the same call under its older name.
    ///
    /// Raises:
    ///     ValueError: the database is closed.
    ///     OSError: writing or syncing the files failed.
    ///     RuntimeError: an internal fault occurred.
    fn save(&self, py: Python<'_>) -> PyResult<()> {
        self.write(py, |s| VectorEngine::persist(&mut s.db).map_err(to_py))
    }

    /// Release the database and its lock without saving. Writes since the
    /// last save() are discarded. Calling close() again does nothing; any
    /// other method then raises ValueError.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.take(py).map(|_| ())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Save on a clean exit, then close; on an exception close without
    /// saving, so a half-finished batch is not made durable
    #[pyo3(signature = (exc_type, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let clean = exc_type.is_none_or(|t| t.is_none());
        if clean && !self.is_closed(py) {
            if let Err(e) = self.save(py) {
                self.take(py)?;
                return Err(e);
            }
        }
        self.take(py)?;
        Ok(false)
    }
}
