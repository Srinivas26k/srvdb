//! Certified and deadline search over RaBitQ codes (ADR 0017)

use super::*;
use crate::quant::rabitq::{RabitqSearch, SearchOutcome, SearchStats, CERTIFIED_CONFIDENCE};
use crate::storage::rabitq::RabitqStorage;
use std::time::Instant;

impl SrvDB {
    /// Search a RaBitQ database with certification and/or a deadline.
    ///
    /// - `certify`: every scanned vector gets an error interval around its
    ///   estimate (ADR 0017). Candidates whose interval clears the k-th
    ///   lower bound are certified members of the top k; only uncertain
    ///   ones are rescored on the full vectors. `stats.confidence` is a
    ///   lower bound on the probability, over the rotation seed, that the
    ///   result is exactly the top k of the scanned vectors.
    /// - `deadline`: the code scan checks the clock between blocks and
    ///   stops when it has passed; results are the best among scanned
    ///   vectors and `stats.completeness` the fraction scanned. The
    ///   rescoring after the scan is not interrupted, so a call can
    ///   overrun the deadline by that work.
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
            self.rabitq_certified(&query.data, k, deadline)?
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
        let rescored = top
            .items
            .iter()
            .map(|(id, _)| Ok((*id, self.exact(vectors, query, *id)?)))
            .collect::<Result<Vec<_>>>()?;
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
    ) -> Result<(Vec<(u64, f32)>, SearchStats)> {
        let (codes, vectors, _) = self.parts()?;
        let (scanned, live) = codes.certified(query, k, &self.deleted, deadline)?;
        let split = scanned.items.into_iter().next().unwrap_or_default();
        let score = |id: &u64| Ok((*id, self.exact(vectors, query, *id)?));
        let uncertain = split
            .uncertain
            .iter()
            .map(score)
            .collect::<Result<Vec<_>>>()?;
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
            confidence: Some(CERTIFIED_CONFIDENCE),
            certified: split.certified.len(),
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
