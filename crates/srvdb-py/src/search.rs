//! `Database.search()`: vector, keyword and hybrid search dispatch
//!
//! Checks how query, text, where, expand, certify and deadline_ms combine,
//! then routes to the matching core search. Goes through `read`
//! (database.rs), so it is safe to call from several Python threads.

use crate::database::PyDatabase;
use crate::errors::{non_negative, to_py};
use crate::graph::{expand_named, to_expansion};
use crate::operations::named;
use crate::where_arg::to_filter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use srvdb::VectorEngine;

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
}
