//! Keyword (BM25) and hybrid (vector + keyword, RRF) search (ADR 0015)

use super::fusion;
use super::tokenize::tokenize;
use crate::engine::validate;
use crate::error::{Error, Result};
use crate::filter::Filter;
use crate::{SearchResult, SrvDB, Vector, VectorEngine};
use std::collections::HashSet;

/// Results taken from each list before fusion: `max(2k, MIN_DEPTH)`
const MIN_DEPTH: usize = 50;

/// Longest accepted query text, in bytes
pub(crate) const MAX_TEXT_BYTES: usize = 64 * 1024;
/// Most distinct words accepted in one query
pub(crate) const MAX_TERMS: usize = 1024;

/// Validate `text` and return its distinct terms, sorted. Bounds the work
/// of one search; runs before any lock is taken.
pub(crate) fn query_terms(text: &str) -> Result<Vec<String>> {
    if text.len() > MAX_TEXT_BYTES {
        return Err(Error::invalid_input(format!(
            "text is {} bytes; keyword queries are limited to {MAX_TEXT_BYTES} bytes",
            text.len()
        )));
    }
    let mut terms = tokenize(text);
    terms.sort();
    terms.dedup();
    if terms.is_empty() {
        return Err(Error::invalid_input(
            "text has no searchable words; pass at least one letter or digit",
        ));
    }
    if terms.len() > MAX_TERMS {
        return Err(Error::invalid_input(format!(
            "text has {} distinct words; keyword queries are limited to {MAX_TERMS}",
            terms.len()
        )));
    }
    Ok(terms)
}

fn not_enabled() -> Error {
    Error::invalid_state("keyword search is not enabled; call enable_keyword_index(field) first")
}

impl SrvDB {
    /// The `k` live records whose indexed field best matches `text` by
    /// BM25 (ADR 0015), best first, with metadata attached. With `filter`,
    /// only records whose metadata matches are considered (reading every
    /// metadata record once). Records that match no word are not returned,
    /// so fewer than `k` results is normal.
    ///
    /// Keyword searches on one database run one at a time (the index may
    /// catch up with metadata during a search); vector searches are not
    /// blocked.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if `k` is 0 or `text` has no letters or
    /// digits, [`Error::InvalidState`] if keyword search is not enabled,
    /// [`Error::Corrupt`] / [`Error::ChecksumMismatch`] if `keyword.index`
    /// is damaged (call [`SrvDB::enable_keyword_index`] to rebuild it), or
    /// an error reading metadata.
    pub fn keyword_search(
        &self,
        text: &str,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        validate::k(k)?;
        let terms = query_terms(text)?;
        let allowed = self.allowed(filter)?;
        let hits = self
            .with_keyword_index(|ix| ix.map(|ix| ix.search(&terms, k, allowed.as_ref())))?
            .ok_or_else(not_enabled)?;
        self.attach(hits)
    }

    /// Hybrid search: the `k` best records by Reciprocal Rank Fusion
    /// (`sum 1 / (60 + rank)`) of vector search for `query` and keyword
    /// search for `text`, each taking its best `max(2k, 50)` results, both
    /// restricted to `filter` when given. The score is the fused score.
    ///
    /// # Errors
    /// As [`SrvDB::keyword_search`], plus [`Error::DimensionMismatch`] /
    /// [`Error::InvalidInput`] for an invalid `query`.
    pub fn search_hybrid(
        &self,
        query: &Vector,
        text: &str,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        validate::k(k)?;
        validate::vector(&query.data, self.dimension)?;
        let terms = query_terms(text)?;
        let depth = k.saturating_mul(2).max(MIN_DEPTH);
        let allowed = self.allowed(filter)?;
        let keyword = self
            .with_keyword_index(|ix| ix.map(|ix| ix.search(&terms, depth, allowed.as_ref())))?
            .ok_or_else(not_enabled)?;
        let vector = match &allowed {
            Some(ids) => self.search_filtered(query, depth, ids)?,
            None => VectorEngine::search(self, query, depth)?,
        };
        let lists = [
            vector.into_iter().map(|r| r.id).collect(),
            keyword.into_iter().map(|(id, _)| id).collect(),
        ];
        self.attach(fusion::rrf(&lists, k))
    }

    fn allowed(&self, filter: Option<&Filter>) -> Result<Option<HashSet<u64>>> {
        filter
            .map(|f| self.ids_where(|meta| f.matches(meta)))
            .transpose()
    }

    fn attach(&self, hits: Vec<(u64, f32)>) -> Result<Vec<SearchResult>> {
        hits.into_iter()
            .map(|(id, s)| Ok(SearchResult::new(id, s, self.metadata_store.get(id)?)))
            .collect()
    }
}
