//! `Database` methods: search, lookup and maintenance
//!
//! Every method goes through `read` or `write` (database.rs), so it is safe
//! to call from several Python threads at once.

use crate::database::{PyDatabase, State};
use crate::errors::{non_negative, to_py};
use crate::graph::{expand_named, to_expansion};
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
    /// Vector, keyword or hybrid search.
    ///
    /// Args:
    ///     query: vector matching the database dimension, finite values, or
    ///         None for a keyword-only search (then `text` is required)
    ///     k: maximum number of results, at least 1
    ///     text: optional keyword query, matched by BM25 against the field
    ///         chosen with enable_keyword_search(). With a query vector the
    ///         two rankings are fused (Reciprocal Rank Fusion).
    ///     where: optional metadata filter, e.g. {"genre": "jazz"} or
    ///         {"year": {"$gte": 1950}}. Operators: $eq, $ne, $gt, $gte,
    ///         $lt, $lte, $in, $nin, $and, $or, $not; "a.b" names a nested
    ///         field. Only matching records are returned.
    ///     expand: optional (link_type, hops) or (link_type, hops,
    ///         max_results): after the k hits, follow outgoing links of
    ///         link_type breadth-first for 1 to 3 hops, adding at most
    ///         max_results records (default 100, at most 1000). Works with
    ///         vector, keyword (text) and hybrid hits; cannot be combined
    ///         with `where` (linked records would not be filtered).
    ///
    ///     certify: RaBitQ modes only. Use the RaBitQ error bounds to
    ///         certify results and rescore only uncertain candidates
    ///         (ADR 0017).
    ///     deadline_ms: RaBitQ modes only. Stop scanning codes after this
    ///         many milliseconds and return the best results found.
    ///
    /// Returns:
    ///     Up to k (id, score) tuples, best first. Scores are cosine
    ///     similarities (approximate in "pq" mode) for vector search, BM25
    ///     scores for keyword-only search and fused RRF scores for hybrid.
    ///     With `expand`, (id, score, hop) tuples instead: the hits with
    ///     hop 0, then linked records ordered by hop, each scored with the
    ///     score of the record it was reached from.
    ///     With certify or deadline_ms, a SearchOutcome: the same tuples
    ///     (iterable, indexable, sliceable, len()) plus completeness,
    ///     confidence, certified, reranked and scanned.
    ///
    /// Raises:
    ///     ValueError: k is 0 or negative, the query has the wrong
    ///         dimension or a NaN or infinite value, `where` is not a valid
    ///         filter, both query and text are None, text has no letters or
    ///         digits, text is given before enable_keyword_search(),
    ///         `expand` is malformed or out of bounds, or `expand` is
    ///         combined with `where`, `certify` or `deadline_ms`;
    ///         certify or deadline_ms on a non-RaBitQ database, without a
    ///         query vector, combined with text or where, or a negative or
    ///         non-finite deadline_ms.
    ///     RuntimeError: an internal fault occurred (including a damaged
    ///         keyword index; enable_keyword_search() rebuilds it).
    ///
    /// A filter reads every metadata record once per call (O(n)). Releases
    /// the GIL; vector searches from several threads run in parallel.
    #[pyo3(signature = (query, k, *, text=None, r#where=None, expand=None, certify=false, deadline_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        py: Python<'_>,
        query: Option<Vec<f32>>,
        k: i64,
        text: Option<String>,
        r#where: Option<Bound<'_, PyAny>>,
        expand: Option<Bound<'_, PyAny>>,
        certify: bool,
        deadline_ms: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        // A certified or deadline search returns a SearchOutcome about the
        // k hits it certified; linked records would carry no such claim
        if expand.is_some() && (certify || deadline_ms.is_some()) {
            return Err(PyValueError::new_err(
                "search() cannot combine `expand` with `certify` or \
                 `deadline_ms`: linked records are not certified; run the \
                 certified search and call neighbors() on its hits instead",
            ));
        }
        if certify || deadline_ms.is_some() {
            let unsupported = if text.is_some() {
                Some("text (keyword and hybrid search)")
            } else if r#where.is_some() {
                Some("where (filtered search)")
            } else {
                None
            };
            if let Some(what) = unsupported {
                return Err(PyValueError::new_err(format!(
                    "certify and deadline_ms work only for a plain vector search; \
                     they cannot be combined with {what}"
                )));
            }
            let Some(query) = query else {
                return Err(PyValueError::new_err(
                    "certify and deadline_ms need a query vector",
                ));
            };
            let outcome = self.search_rabitq(py, query, k, certify, deadline_ms)?;
            return Ok(Py::new(py, outcome)?.into_any());
        }
        let k = non_negative("k", k)?;
        // Linked records would bypass the filter, so `where` and `expand`
        // together are refused rather than given a surprising meaning
        if expand.is_some() && r#where.is_some() {
            return Err(PyValueError::new_err(
                "search() cannot combine `where` and `expand`: linked records \
                 are not filtered; filter the hits and call neighbors() instead",
            ));
        }
        let expansion = expand.map(|e| to_expansion(&e)).transpose()?;
        let filter = r#where.map(|w| to_filter(py, &w)).transpose()?;
        let query = query.map(srvdb::Vector::new);
        let (hits, expanded) = self.read(py, |s| {
            let results = match (&query, &text, &filter) {
                (None, None, _) => {
                    return Err(PyValueError::new_err("pass a query vector, text, or both"))
                }
                (Some(q), Some(t), f) => s.db.search_hybrid(q, t, k, f.as_ref()),
                (None, Some(t), f) => s.db.keyword_search(t, k, f.as_ref()),
                (Some(q), None, Some(f)) => s.db.search_where(q, k, f),
                (Some(q), None, None) => VectorEngine::search(&s.db, q, k),
            }
            .map_err(to_py)?;
            match &expansion {
                Some(e) => Ok((Vec::new(), Some(expand_named(s, &results, e)?))),
                None => Ok((named(s, results), None)),
            }
        })?;
        Ok(match expanded {
            Some(e) => e.into_pyobject(py)?.into_any().unbind(),
            None => hits.into_pyobject(py)?.into_any().unbind(),
        })
    }

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
}
