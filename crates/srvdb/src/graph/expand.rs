//! k-hop search expansion along links of one type (ADR 0016)

use super::{validate_type, LinkGraph};
use crate::error::{Error, Result};
use crate::SearchResult;
use std::collections::HashSet;

/// How [`crate::SrvDB::search_expand`] follows links from the search hits.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Expansion {
    /// Link type to follow (outgoing links only)
    pub link_type: String,
    /// Number of hops, 1 to [`Expansion::MAX_HOPS`]
    pub hops: u32,
    /// Most expanded records added after the `k` hits, 1 to
    /// [`Expansion::MAX_RESULTS`]
    pub max_results: usize,
}

impl Expansion {
    /// Largest allowed `hops`
    pub const MAX_HOPS: u32 = 3;
    /// Largest allowed `max_results`
    pub const MAX_RESULTS: usize = 1000;
    /// `max_results` used by [`Expansion::new`]
    pub const DEFAULT_RESULTS: usize = 100;

    /// Follow `link_type` links for `hops` hops, adding at most
    /// [`Expansion::DEFAULT_RESULTS`] records. Checked when searching.
    pub fn new(link_type: impl Into<String>, hops: u32) -> Self {
        Self {
            link_type: link_type.into(),
            hops,
            max_results: Self::DEFAULT_RESULTS,
        }
    }

    /// Set the most expanded records added (checked when searching)
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_type(&self.link_type)?;
        if !(1..=Self::MAX_HOPS).contains(&self.hops) {
            return Err(Error::invalid_input(format!(
                "hops must be 1 to {}, got {}",
                Self::MAX_HOPS,
                self.hops
            )));
        }
        if !(1..=Self::MAX_RESULTS).contains(&self.max_results) {
            return Err(Error::invalid_input(format!(
                "max_results must be 1 to {}, got {}",
                Self::MAX_RESULTS,
                self.max_results
            )));
        }
        Ok(())
    }
}

/// A search hit (hop 0) or a record reached by following links.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExpandedResult {
    /// Internal record id
    pub id: u64,
    /// Similarity for a hit; for an expanded record, the score of the
    /// record it was reached from (inherited, not its own similarity)
    pub score: f32,
    /// 0 for a search hit, otherwise the number of links followed
    pub hop: u32,
    /// The record's metadata
    pub metadata: Option<String>,
}

/// Breadth-first expansion from `hits`. Returns the hits (hop 0) followed
/// by up to `max_results` (id, inherited score, hop) records, ordered by
/// hop and then score. Each record appears once, at its smallest hop.
pub(crate) fn expand(
    graph: &LinkGraph,
    hits: &[SearchResult],
    expansion: &Expansion,
) -> Vec<(u64, f32, u32)> {
    let mut seen: HashSet<u64> = hits.iter().map(|r| r.id).collect();
    let mut out: Vec<(u64, f32, u32)> = hits.iter().map(|r| (r.id, r.score, 0)).collect();
    let mut frontier: Vec<(u64, f32)> = hits.iter().map(|r| (r.id, r.score)).collect();
    let mut added = 0;
    for hop in 1..=expansion.hops {
        // Best parents first, so a record inherits the best score at this hop
        frontier.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut next = Vec::new();
        'level: for &(id, score) in &frontier {
            for target in graph.targets(id, &expansion.link_type) {
                if added == expansion.max_results {
                    break 'level;
                }
                if seen.insert(target) {
                    next.push((target, score));
                    added += 1;
                }
            }
        }
        out.extend(next.iter().map(|&(id, score)| (id, score, hop)));
        if next.is_empty() || added == expansion.max_results {
            break;
        }
        frontier = next;
    }
    out
}
