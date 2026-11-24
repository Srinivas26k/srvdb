//! Filtered search: k nearest neighbours among an allowed set of ids
//!
//! With an HNSW index the filter is applied during graph traversal (see
//! index::hnsw::filtered); every other mode, and HNSW with a very
//! selective filter, scores exactly the allowed ids.

use super::*;
use rayon::prelude::*;
use std::collections::HashSet;

/// Exact scan of the matches beats the filtered graph walk when
/// `matches^2 < EXACT_COST_RATIO * ef * M * live`.
///
/// Measured (docs/benchmarks.md, "Filtered search"): the filtered walk
/// visits about `ef * M / s` nodes at selectivity `s` (12.5k of 20k at 10%,
/// 4.5k at 50%, ef = 100, M = 16) at about 2 us per visit, while scoring
/// one matching id exactly costs about 0.13 us. Equal cost is
/// `s * n * 0.13 = 2 * ef * M / s`, i.e. `s^2 * n = 15 * ef * M`. With the
/// defaults (ef 50, M 16) that is below 35% matching at 100k vectors, 11%
/// at 1M and 3.5% at 10M: always above the ~1% where the walk degrades.
pub(crate) const EXACT_COST_RATIO: f64 = 15.0;

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
    /// is cheaper (see `EXACT_COST_RATIO`: under about 11% of 1M vectors,
    /// 35% of 100k), or the graph walk finds fewer than `k`, the allowed
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
