//! Certified and deadline search over RaBitQ codes (ADR 0017)

use super::*;
use crate::quant::rabitq::{
    RabitqSearch, SearchOutcome, SearchStats, CERTIFIED_CONFIDENCE, DEFAULT_MAX_RERANK,
};
use crate::storage::rabitq::RabitqStorage;
use std::time::Instant;

impl SrvDB {
    /// Search a RaBitQ database with certification and/or a deadline.
    ///
    /// - `certify`: every scanned vector gets an error interval around its
    ///   estimate (ADR 0017). Candidates whose interval clears the k-th
    ///   lower bound are certified members of the top k; only uncertain
    ///   ones are rescored on the full vectors, at most
    ///   `options.max_rerank` of them (by default
    ///   `max(k * rerank, DEFAULT_MAX_RERANK)`). `stats.confidence` is a
    ///   nominal, approximate lower bound on the probability, over a
    ///   hypothetical random rotation, that the result is exactly the top
    ///   k of the scanned vectors; see [`SearchStats::confidence`] for
    ///   what it does not mean. It is `None` if the cap or the deadline
    ///   cut the rerank short.
    /// - `deadline`: the code scan checks the clock between blocks and
    ///   stops when it has passed; results are the best among scanned
    ///   vectors and `stats.completeness` the fraction scanned. The
    ///   rescoring also checks the clock between batches of 256.
    ///
    /// With neither option this returns what `search` returns. Results
    /// carry metadata like `search`. Takes `&self`; safe to call from
    /// several threads.
    ///
    /// # Errors
    /// `InvalidInput` if this is not a RaBitQ database, `k` is 0 or the
    /// query has the wrong dimension or a non-finite value; I/O errors
    /// reading metadata.
    pub fn search_rabitq_with(
        &self,
        query: &Vector,
        k: usize,
        options: &RabitqSearch,
    ) -> Result<SearchOutcome> {
        if self.rabitq.is_none() {
            return Err(Error::invalid_input(
                "certified and deadline search need a RaBitQ database; \
                 create it with mode 'rabitq1', 'rabitq2' or 'rabitq4'",
            ));
        }
        validate::k(k)?;
        validate::vector(&query.data, self.dimension)?;
        let deadline = options.deadline.and_then(|d| Instant::now().checked_add(d));
        let (raw, stats) = if options.certify {
            self.rabitq_certified(&query.data, k, deadline, options.max_rerank)?
        } else {
            self.rabitq_top(&query.data, k, deadline)?
        };
        let results = raw
            .into_iter()
            .map(|(id, score)| {
                let meta = self.metadata_store.get(id)?;
                Ok(SearchResult::new(id, score, meta))
            })
            .collect::<Result<_>>()?;
        Ok(SearchOutcome { results, stats })
    }

    fn parts(&self) -> Result<(&RabitqStorage, &storage::flat::VectorStorage, usize)> {
        let (Some(codes), Some(config), Some(vectors)) =
            (&self.rabitq, &self.rabitq_config, &self.vector_storage)
        else {
            return Err(Error::invalid_state("RaBitQ storage not initialized"));
        };
        Ok((codes, vectors, config.rerank))
    }

    /// Exact cosine of `query` and stored vector `id`
    fn exact(&self, vectors: &storage::flat::VectorStorage, query: &[f32], id: u64) -> Result<f32> {
        // sync_rabitq keeps codes.count() == vectors.count(), so every
        // candidate id has a vector; a miss is an internal fault
        let v = vectors.get(id).ok_or_else(|| {
            Error::Internal(format!("RaBitQ candidate {id} has no stored vector"))
        })?;
        Ok(search::cosine_similarity(query, v))
    }

    /// Rescore `ids` exactly in batches, checking `deadline` between
    /// batches: (scores, whether every id was rescored)
    fn rescore(
        &self,
        vectors: &storage::flat::VectorStorage,
        query: &[f32],
        ids: &[u64],
        deadline: Option<Instant>,
    ) -> Result<(Vec<(u64, f32)>, bool)> {
        /// Vectors rescored between deadline checks
        const BATCH: usize = 256;
        let mut out = Vec::with_capacity(ids.len());
        for (b, batch) in ids.chunks(BATCH).enumerate() {
            if b > 0 && deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok((out, false));
            }
            for &id in batch {
                out.push((id, self.exact(vectors, query, id)?));
            }
        }
        Ok((out, true))
    }

    /// Scan for `k * rerank` candidates and rescore them
    pub(super) fn rabitq_top(
        &self,
        query: &[f32],
        k: usize,
        deadline: Option<Instant>,
    ) -> Result<(Vec<(u64, f32)>, SearchStats)> {
        let (codes, vectors, rerank) = self.parts()?;
        let live = codes.live(&self.deleted);
        let stats = |scanned: usize, reranked: usize| SearchStats {
            completeness: fraction(scanned, live),
            confidence: None,
            certified: 0,
            reranked,
            scanned,
        };
        if rerank == 0 {
            let top = codes.top(query, k, &self.deleted, deadline);
            return Ok((top.items, stats(top.scanned, 0)));
        }
        let top = codes.top(query, k.saturating_mul(rerank), &self.deleted, deadline);
        let ids: Vec<u64> = top.items.iter().map(|(id, _)| *id).collect();
        let (rescored, _) = self.rescore(vectors, query, &ids, deadline)?;
        let reranked = rescored.len();
        Ok((
            search::select_top_k(rescored, k),
            stats(top.scanned, reranked),
        ))
    }

    fn rabitq_certified(
        &self,
        query: &[f32],
        k: usize,
        deadline: Option<Instant>,
        max_rerank: Option<usize>,
    ) -> Result<(Vec<(u64, f32)>, SearchStats)> {
        let (codes, vectors, rerank) = self.parts()?;
        let cap = max_rerank
            .unwrap_or_else(|| k.saturating_mul(rerank).max(DEFAULT_MAX_RERANK))
            .max(1);
        let (scanned, live) = codes.certified(query, k, &self.deleted, deadline)?;
        let split = scanned.items.into_iter().next().unwrap_or_default();
        let score = |id: &u64| Ok((*id, self.exact(vectors, query, *id)?));
        // Uncertain ids come best upper bound first, so a cap keeps the
        // most promising ones
        let capped = split.uncertain.len() > cap;
        let wanted = &split.uncertain[..split.uncertain.len().min(cap)];
        let (uncertain, complete) = self.rescore(vectors, query, wanted, deadline)?;
        let certified = !capped && complete;
        let reranked = uncertain.len();
        let rest = search::select_top_k(uncertain, k - split.certified.len());
        // Certified members are scored exactly too (at most k reads) so
        // scores and order match every other mode
        let mut all = split
            .certified
            .iter()
            .map(score)
            .collect::<Result<Vec<_>>>()?;
        all.extend(rest);
        let stats = SearchStats {
            completeness: fraction(scanned.scanned, live),
            confidence: certified.then_some(CERTIFIED_CONFIDENCE),
            // A cut-short rerank certifies nothing: report 0, not a partial count
            certified: if certified { split.certified.len() } else { 0 },
            reranked,
            scanned: scanned.scanned,
        };
        Ok((search::select_top_k(all, k), stats))
    }
}

fn fraction(scanned: usize, live: usize) -> f64 {
    if live == 0 {
        1.0
    } else {
        (scanned as f64 / live as f64).min(1.0)
    }
}

#[cfg(test)]
#[path = "rabitq_search_tests.rs"]
mod tests;
