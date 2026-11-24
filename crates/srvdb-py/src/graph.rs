//! `Database` link methods and search expansion (ADR 0016)

use crate::database::{PyDatabase, State};
use crate::errors::{non_negative, to_py};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Internal id of a live string id, or `ValueError`
fn internal(state: &State, id: &str) -> PyResult<u64> {
    state
        .id_map
        .get(id)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("unknown id '{id}'")))
}

/// Parse `expand=(type, hops)` or `(type, hops, max_results)`
pub(crate) fn to_expansion(value: &Bound<'_, PyAny>) -> PyResult<srvdb::Expansion> {
    let bad = || {
        PyValueError::new_err(
            "expand must be a tuple (link_type, hops) or (link_type, hops, max_results)",
        )
    };
    let (link_type, hops, max) = if let Ok((t, h)) = value.extract::<(String, i64)>() {
        (t, h, None)
    } else if let Ok((t, h, m)) = value.extract::<(String, i64, i64)>() {
        (t, h, Some(m))
    } else {
        return Err(bad());
    };
    let hops = u32::try_from(hops)
        .map_err(|_| PyValueError::new_err(format!("hops must be 1 to 3, got {hops}")))?;
    let mut expansion = srvdb::Expansion::new(link_type, hops);
    if let Some(max) = max {
        expansion = expansion.with_max_results(non_negative("max_results", max)?);
    }
    Ok(expansion)
}

/// Run an expanded search; results without a string id are dropped
pub(crate) fn search_expanded(
    state: &State,
    query: &srvdb::Vector,
    k: usize,
    expansion: &srvdb::Expansion,
) -> PyResult<Vec<(String, f32, u32)>> {
    let results = state.db.search_expand(query, k, expansion).map_err(to_py)?;
    Ok(results
        .into_iter()
        .filter_map(|r| {
            let id = state.reverse_id_map.get(&r.id)?;
            Some((id.clone(), r.score, r.hop))
        })
        .collect())
}

#[pymethods]
impl PyDatabase {
    /// Add the directed link `source -link_type-> target`.
    ///
    /// Visible at once; durable after the next save(), atomically with the
    /// records saved with it. Deleting either record removes the link;
    /// upsert() keeps it on the new version.
    ///
    /// Returns:
    ///     True if the link is new, False if it already existed.
    ///
    /// Raises:
    ///     ValueError: an id is unknown, source equals target, link_type
    ///         does not match [a-z][a-z0-9_]{0,63}, or source already has
    ///         1024 outgoing links.
    fn link(
        &self,
        py: Python<'_>,
        source: String,
        link_type: String,
        target: String,
    ) -> PyResult<bool> {
        self.write(py, |s| {
            let (a, b) = (internal(s, &source)?, internal(s, &target)?);
            s.db.link(a, &link_type, b).map_err(to_py)
        })
    }

    /// Remove the link `source -link_type-> target`. Durable after the
    /// next save().
    ///
    /// Returns:
    ///     True if the link existed (False also for unknown ids).
    ///
    /// Raises:
    ///     ValueError: link_type is not a valid link type name.
    fn unlink(
        &self,
        py: Python<'_>,
        source: String,
        link_type: String,
        target: String,
    ) -> PyResult<bool> {
        self.write(py, |s| {
            // An unknown id has no links; u64::MAX is never a record id, so
            // the core still validates link_type and reports "no such link"
            let a = s.id_map.get(&source).copied().unwrap_or(u64::MAX);
            let b = s.id_map.get(&target).copied().unwrap_or(u64::MAX);
            s.db.unlink(a, &link_type, b).map_err(to_py)
        })
    }

    /// Outgoing links of `id` as (link_type, target id) tuples, sorted;
    /// only links of `type` if given.
    ///
    /// Raises:
    ///     ValueError: the id is unknown or `type` is not a valid name.
    #[pyo3(signature = (id, r#type=None))]
    fn neighbors(
        &self,
        py: Python<'_>,
        id: String,
        r#type: Option<String>,
    ) -> PyResult<Vec<(String, String)>> {
        self.read(py, |s| {
            let links =
                s.db.neighbors(internal(s, &id)?, r#type.as_deref())
                    .map_err(to_py)?;
            Ok(links
                .into_iter()
                .filter_map(|(t, target)| Some((t, s.reverse_id_map.get(&target)?.clone())))
                .collect())
        })
    }
}
