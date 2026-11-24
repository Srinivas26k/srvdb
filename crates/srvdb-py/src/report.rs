//! `Database` reports: `count()`, `info()`, `repr()` and `memory_report()`
//! (ADR 0018)

use crate::database::PyDatabase;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pymethods]
impl PyDatabase {
    /// Number of searchable vectors (deleted ones are not counted)
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        self.read(py, |s| Ok(s.db.live_count() as usize))
    }

    /// Estimated resident memory, per part, and the memory budget.
    ///
    /// Returns:
    ///     A dict with "mode", "bits" (RaBitQ bits or None), "dimension",
    ///     "count" (records, deleted ones included), "vector_bytes",
    ///     "index_bytes", "links_bytes", "keyword_bytes", "total_bytes",
    ///     "budget_bytes", "limit_bytes" and "capacity" (None without a
    ///     memory budget) and "exact" (True for flat search).
    ///
    /// Raises:
    ///     ValueError: the database is closed.
    fn memory_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.read(py, |s| Ok(s.db.memory_report()))?;
        let d = PyDict::new(py);
        d.set_item("mode", r.mode)?;
        d.set_item("bits", r.bits)?;
        d.set_item("dimension", r.dimension)?;
        d.set_item("count", r.count)?;
        d.set_item("vector_bytes", r.vector_bytes)?;
        d.set_item("index_bytes", r.index_bytes)?;
        d.set_item("links_bytes", r.links_bytes)?;
        d.set_item("keyword_bytes", r.keyword_bytes)?;
        d.set_item("total_bytes", r.total_bytes)?;
        d.set_item("budget_bytes", r.budget_bytes)?;
        d.set_item("limit_bytes", r.limit_bytes)?;
        d.set_item("capacity", r.capacity)?;
        d.set_item("exact", r.exact)?;
        Ok(d)
    }

    /// One-line summary: dimension, mode and vector count (plus memory and
    /// compression in "pq" mode). Also used as repr().
    fn info(&self, py: Python<'_>) -> PyResult<String> {
        self.read(py, |s| {
            let stats = match s.db.get_stats() {
                Some(stats) => format!(
                    "vectors={}, memory={}MB, compression={:.1}x",
                    stats.vector_count,
                    stats.memory_bytes / 1024 / 1024,
                    stats.compression_ratio
                ),
                None => format!("vectors={}", s.db.live_count()),
            };
            // A memory budget can change the mode on add (ADR 0018)
            let mode = match s.db.memory_report().budget_bytes {
                Some(_) => s.db.index_name(),
                None => &s.index_type,
            };
            Ok(format!(
                "SrvDB(dimension={}, mode='{mode}', {stats})",
                s.dimension
            ))
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if self.is_closed(py) {
            return Ok("SrvDB(closed)".to_string());
        }
        self.info(py)
    }
}
