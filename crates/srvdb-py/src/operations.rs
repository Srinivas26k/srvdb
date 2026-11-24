//! `Database` methods: lookup and maintenance (search() is in search.rs)
//!
//! Every method goes through `read` or `write` (database.rs), so it is safe
//! to call from several Python threads at once.

use crate::database::{PyDatabase, State};
use crate::errors::{non_negative, to_py};
use crate::where_arg::to_filter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use srvdb::{SearchResult, VectorEngine};

/// Translate internal ids to string ids, dropping entries without one
pub(crate) fn named(state: &State, results: Vec<SearchResult>) -> Vec<(String, f32)> {
    results
        .into_iter()
        .filter_map(|r| {
            state
                .reverse_id_map
                .get(&r.id)
                .map(|id| (id.clone(), r.score))
        })
        .collect()
}

#[pymethods]
impl PyDatabase {
    /// Index the string metadata field `field` for keyword (BM25) search
    /// with search(text=...). "a.b" names a nested field; a list of
    /// strings is indexed as their concatenation. Reuses the saved index
    /// when it covers the same field, otherwise builds it from metadata.
    /// The index is saved by the next save().
    ///
    /// Raises:
    ///     ValueError: field is empty or malformed.
    ///     OSError: reading metadata failed.
    #[pyo3(signature = (field="text"))]
    fn enable_keyword_search(&self, py: Python<'_>, field: &str) -> PyResult<()> {
        self.write(py, |s| s.db.enable_keyword_index(field).map_err(to_py))
    }

    /// Search for several queries at once.
    ///
    /// Returns one result list per query, in query order. Arguments, scores
    /// and errors as for search(); a `where` filter is evaluated once for
    /// the whole batch. Releases the GIL like search().
    #[pyo3(signature = (queries, k, *, r#where=None))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<f32>>,
        k: i64,
        r#where: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Vec<(String, f32)>>> {
        let k = non_negative("k", k)?;
        let filter = r#where.map(|w| to_filter(py, &w)).transpose()?;
        let queries: Vec<srvdb::Vector> = queries.into_iter().map(srvdb::Vector::new).collect();
        self.read(py, |s| {
            let batches = match &filter {
                Some(f) => s.db.search_batch_where(&queries, k, f),
                None => VectorEngine::search_batch(&s.db, &queries, k),
            }
            .map_err(to_py)?;
            Ok(batches.into_iter().map(|b| named(s, b)).collect())
        })
    }

    /// Metadata stored for `id`, as a JSON object string that also holds
    /// the id under the key "__id__", or None if no live vector has this
    /// id (like dict.get).
    ///
    /// Raises:
    ///     OSError: reading the metadata failed.
    ///     RuntimeError: an internal fault occurred.
    fn get(&self, py: Python<'_>, id: String) -> PyResult<Option<String>> {
        self.read(py, |s| {
            let Some(internal_id) = s.id_map.get(&id) else {
                return Ok(None);
            };
            VectorEngine::get_metadata(&s.db, *internal_id).map_err(to_py)
        })
    }

    /// Number of searchable vectors (deleted ones are not counted)
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        self.read(py, |s| Ok(s.db.live_count() as usize))
    }

    /// Alias of save(): make everything written so far durable.
    ///
    /// When it returns, every earlier add, upsert and delete survives a
    /// crash or power loss; writes after the last save() may be lost.
    ///
    /// Raises:
    ///     OSError: writing or syncing the files failed.
    ///     RuntimeError: an internal fault occurred.
    fn persist(&self, py: Python<'_>) -> PyResult<()> {
        self.write(py, |s| VectorEngine::persist(&mut s.db).map_err(to_py))
    }

    /// Every live string id, in no particular order
    fn get_all_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.read(py, |s| Ok(s.id_map.keys().cloned().collect()))
    }

    /// Switch the index mode: 'flat', 'hnsw', 'sq8' (or 'scalar'), 'ivf' or
    /// 'auto' (case-insensitive). Recorded by the next persist().
    ///
    /// Raises:
    ///     ValueError: unknown mode (including 'pq', which needs
    ///         new_product_quantized()).
    fn set_mode(&self, py: Python<'_>, mode: String) -> PyResult<()> {
        let index_mode = match mode.to_lowercase().as_str() {
            "flat" => srvdb::IndexMode::Flat,
            "hnsw" => srvdb::IndexMode::Hnsw,
            "sq8" | "scalar" => srvdb::IndexMode::Sq8,
            "ivf" => srvdb::IndexMode::Ivf,
            "auto" => srvdb::IndexMode::Auto,
            _ => return Err(PyValueError::new_err(format!("Invalid mode: {mode}"))),
        };
        self.write(py, |s| {
            s.db.set_mode(index_mode);
            s.index_type = mode;
            Ok(())
        })
    }

    /// Set HNSW's search candidate list size: higher gives better recall,
    /// lower is faster (typical 50-300). Saved by persist().
    ///
    /// Raises:
    ///     ValueError: ef_search is 0, negative or above 10000.
    fn set_ef_search(&self, py: Python<'_>, ef_search: i64) -> PyResult<()> {
        let ef_search = non_negative("ef_search", ef_search)?;
        self.write(py, |s| s.db.set_ef_search(ef_search).map_err(to_py))
    }

    /// Configure IVF parameters
    ///
    /// Args:
    ///     nlist: Number of partitions (centroids)
    ///     nprobe: Number of partitions to search (search quality vs speed)
    ///
    /// Call before set_mode("ivf") so the index is created with these
    /// values; in "ivf" mode it replaces the index with an untrained one.
    /// Either way, call train_ivf() afterwards.
    #[pyo3(signature = (nlist=100, nprobe=10))]
    fn configure_ivf(&self, py: Python<'_>, nlist: i64, nprobe: i64) -> PyResult<()> {
        let (nlist, nprobe) = (
            non_negative("nlist", nlist)?,
            non_negative("nprobe", nprobe)?,
        );
        let config = srvdb::ivf::IVFConfig {
            nlist,
            nprobe,
            max_iterations: 10,
            tolerance: 0.001,
            hnsw_config: Default::default(),
        };
        self.write(py, |s| s.db.configure_ivf(config).map_err(to_py))
    }

    /// Train the IVF index on the stored vectors (releases the GIL)
    ///
    /// Raises:
    ///     ValueError: the database is not in "ivf" mode; call
    ///         set_mode("ivf") first.
    fn train_ivf(&self, py: Python<'_>) -> PyResult<()> {
        self.write(py, |s| s.db.train_ivf().map_err(to_py))
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
            Ok(format!(
                "SrvDB(dimension={}, mode='{}', {stats})",
                s.dimension, s.index_type
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
