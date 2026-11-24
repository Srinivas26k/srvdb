//! Public options and results of certified and deadline search (ADR 0017)

use crate::SearchResult;
use std::time::Duration;

/// Options for [`crate::SrvDB::search_rabitq_with`].
///
/// Build with [`RabitqSearch::new`] and the `with_*` methods; the struct
/// is `#[non_exhaustive]` so options may be added later.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RabitqSearch {
    /// Use the RaBitQ error bounds: certify candidates whose interval
    /// clears the k-th lower bound and rescore only the uncertain ones
    pub certify: bool,
    /// Stop scanning code blocks once this much time has passed
    pub deadline: Option<Duration>,
}

impl RabitqSearch {
    /// A plain search: no certification, no deadline
    pub fn new() -> Self {
        Self::default()
    }

    /// Turn certification on or off
    pub fn with_certify(mut self, certify: bool) -> Self {
        self.certify = certify;
        self
    }

    /// Set a deadline, measured from the start of the call
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// How a search went, next to its results
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SearchStats {
    /// Fraction of live vectors whose codes were scanned, in `[0, 1]`;
    /// below 1 only when the deadline passed
    pub completeness: f64,
    /// Certified search only: lower bound on the probability, over the
    /// rotation seed, that the results are exactly the top k of the
    /// scanned vectors (ADR 0017). `None` for an uncertified search.
    pub confidence: Option<f64>,
    /// Results whose membership was certified without exact rescoring
    pub certified: usize,
    /// Candidates rescored against the full vectors
    pub reranked: usize,
    /// Live vectors whose codes were scanned
    pub scanned: usize,
}

/// Results of [`crate::SrvDB::search_rabitq_with`] with their statistics
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SearchOutcome {
    /// Up to `k` results, best first, with exact cosine scores unless
    /// rerank is 0 and certification is off
    pub results: Vec<SearchResult>,
    /// Completeness, confidence and work done
    pub stats: SearchStats,
}
