//! Filtered search: k nearest neighbours among an allowed set of ids
//!
//! With an HNSW index the filter is applied during graph traversal (see
//! index::hnsw::filtered); every other mode, and HNSW with a very
//! selective filter, scores exactly the allowed ids.

use super::*;
use crate::filter::Filter;
use rayon::prelude::*;
use std::collections::HashSet;

/// Exact scan of the matches beats the filtered graph walk when
/// `matches^2 < EXACT_COST_RATIO * ef * M * live`.
///
/// Equal cost is `s * n * c_exact = c_walk(s)` at selectivity `s`, which
/// gives `s^2 * n = ratio * ef * M`. The ratio is measured by the ignored
/// test `filtered_walk_cost` (index/hnsw/cost_tests.rs); the numbers, the
/// machine and the command are in docs/benchmarks.md, "Filtered search".
pub(crate) const EXACT_COST_RATIO: f64 = 4.0;

/// True when scoring `matches` ids exactly is cheaper than the graph walk
pub(crate) fn prefer_exact(matches: usize, live: u64, ef: usize, m: usize) -> bool {
    let matches = matches as f64;
    matches * matches < EXACT_COST_RATIO * (ef * m) as f64 * live as f64
}

impl SrvDB {
    /// Search for the `k` nearest neighbours of `query` among `allowed`.
    ///
    /// Ids that are deleted or not in the database are ignored. Always
    /// returns `min(k, number of live allowed ids)` results, best first,
    /// with metadata attached. With an HNSW index the filter is evaluated
    /// during traversal; when few enough vectors match that scoring them
    /// is cheaper (`matches^2 < 4 * ef * M * live`, measured in
    /// docs/benchmarks.md), or the graph walk finds fewer than `k`, the allowed
    /// ids are scored exactly. Other modes always score the allowed ids exactly.
    ///
    /// Build `allowed` with [`SrvDB::ids_where`] and reuse it across
    /// queries while the filter and the data do not change.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if `k` is 0 or `query` contains NaN or
    /// infinity, [`Error::DimensionMismatch`] if its length differs from
    /// the database dimension, and an internal error if storage is not
    /// initialized or metadata cannot be read.
    pub fn search_filtered(
        &self,
        query: &Vector,
        k: usize,
        allowed: &HashSet<u64>,
    ) -> Result<Vec<SearchResult>> {
        validate::k(k)?;
        validate::vector(&query.data, self.dimension)?;
        let count = self.count();
        let live: Vec<u64> = allowed
            .iter()
            .copied()
            .filter(|id| *id < count && !self.deleted.contains(id))
            .collect();
        let score = self.scorer(&query.data)?;
        let live_total = count.saturating_sub(self.deleted.len() as u64);

        let mut raw = Vec::new();
        let graph = self.hnsw_index.as_ref().filter(|h| {
            let ef = h.ef_search().max(k);
            !prefer_exact(live.len(), live_total, ef, h.m())
        });
        if let Some(hnsw) = graph {
            let accept = |id: u64| allowed.contains(&id) && !self.deleted.contains(&id);
            // Unscorable ids (past the end of storage) cannot be returned,
            // so treating them as infinitely far only stops them guiding
            let distance = |_: u64, id: u64| score(id).map_or(f32::MAX, |s| 1.0 - s);
            raw = hnsw
                .search_accepted(0, k, &distance, &accept)?
                .into_iter()
                .map(|(id, d)| (id, 1.0 - d))
                .collect();
        }
        if raw.len() < k.min(live.len()) {
            let scored: Vec<(u64, f32)> = live
                .par_iter()
                .filter_map(|&id| score(id).map(|s| (id, s)))
                .collect();
            raw = search::select_top_k(scored, k);
        }
        raw.into_iter()
            .map(|(id, s)| Ok(SearchResult::new(id, s, self.metadata_store.get(id)?)))
            .collect()
    }

    /// Ids of live vectors whose metadata JSON satisfies `predicate`.
    ///
    /// Reads every metadata record once (O(n)); keep the result and pass
    /// it to [`SrvDB::search_filtered`] for many queries.
    ///
    /// # Errors
    /// An internal error if the metadata store cannot be read.
    pub fn ids_where<P>(&self, predicate: P) -> Result<HashSet<u64>>
    where
        P: Fn(&str) -> bool,
    {
        Ok(self
            .metadata_store
            .all()?
            .into_iter()
            .filter(|(id, meta)| !self.deleted.contains(id) && predicate(meta))
            .map(|(id, _)| id)
            .collect())
    }

    /// Search for the `k` nearest neighbours of `query` among the live
    /// vectors whose metadata matches `filter`.
    ///
    /// Same as [`SrvDB::ids_where`] with [`Filter::matches`] followed by
    /// [`SrvDB::search_filtered`]: reads every metadata record once (O(n))
    /// and then returns `min(k, matches)` results, best first. To run many
    /// queries with one filter, use [`SrvDB::search_batch_where`] or build
    /// the id set yourself.
    ///
    /// # Errors
    /// As [`SrvDB::search_filtered`] and [`SrvDB::ids_where`].
    pub fn search_where(
        &self,
        query: &Vector,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        validate::k(k)?;
        validate::vector(&query.data, self.dimension)?;
        let allowed = self.ids_where(|meta| filter.matches(meta))?;
        log::debug!("filtered search: {} matching ids", allowed.len());
        self.search_filtered(query, k, &allowed)
    }

    /// [`SrvDB::search_where`] for several queries, evaluating `filter`
    /// once. Returns one result list per query, in query order.
    ///
    /// # Errors
    /// As [`SrvDB::search_where`]; every query is validated before the
    /// metadata is read.
    pub fn search_batch_where(
        &self,
        queries: &[Vector],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Vec<SearchResult>>> {
        validate::k(k)?;
        for query in queries {
            validate::vector(&query.data, self.dimension)?;
        }
        let allowed = self.ids_where(|meta| filter.matches(meta))?;
        queries
            .iter()
            .map(|q| self.search_filtered(q, k, &allowed))
            .collect()
    }

    /// Similarity of `query` to one stored vector, using the same storage
    /// (and therefore the same approximation) as unfiltered search.
    fn scorer<'a>(
        &'a self,
        query: &'a [f32],
    ) -> Result<Box<dyn Fn(u64) -> Option<f32> + Sync + 'a>> {
        if let Some(ref scalar) = self.scalar_storage {
            return Ok(Box::new(move |id| {
                scalar.get(id).map(|v| search::cosine_similarity(query, &v))
            }));
        }
        if self.config.enabled {
            let Some(ref q) = self.quantized_storage else {
                // Quantizer not trained yet: nothing has been added
                return Ok(Box::new(|_| None));
            };
            let table = q.quantizer.compute_distance_table(query);
            return Ok(Box::new(move |id| {
                q.get(id)
                    .map(|code| q.quantizer.asymmetric_distance(code, &table))
            }));
        }
        let Some(ref v) = self.vector_storage else {
            return Err(Error::invalid_state("Vector storage not initialized"));
        };
        Ok(Box::new(move |id| {
            v.get(id).map(|x| search::cosine_similarity(query, x))
        }))
    }
}
