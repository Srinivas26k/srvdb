//! In-memory inverted index and BM25 scoring (ADR 0015)

use super::tokenize::tokenize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

/// BM25 term-frequency saturation
pub(crate) const K1: f64 = 1.2;
/// BM25 length normalisation
pub(crate) const B: f64 = 0.75;

/// Inverted index over one metadata field
#[derive(Debug, Clone, Default)]
pub(crate) struct InvertedIndex {
    /// Metadata field the index covers
    pub(crate) field: String,
    /// Records `0..covered` have been considered for indexing
    pub(crate) covered: u64,
    /// term -> (id -> term frequency)
    pub(crate) postings: BTreeMap<String, BTreeMap<u64, u32>>,
    /// id -> number of tokens, for every indexed record
    pub(crate) lengths: BTreeMap<u64, u32>,
    /// Sum of `lengths`
    pub(crate) total_len: u64,
    /// id -> its distinct terms (in memory only, derived from `postings`)
    pub(crate) forward: HashMap<u64, Vec<String>>,
}

impl InvertedIndex {
    /// An empty index for `field` that has seen no records
    pub(crate) fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            ..Default::default()
        }
    }

    /// Number of indexed records
    pub(crate) fn docs(&self) -> usize {
        self.lengths.len()
    }

    /// Index `text` as record `id` (replacing any earlier entry)
    pub(crate) fn add(&mut self, id: u64, text: &str) {
        self.remove(id);
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }
        let mut tf: HashMap<String, u32> = HashMap::new();
        for token in tokens.iter() {
            *tf.entry(token.clone()).or_insert(0) += 1;
        }
        let mut distinct = Vec::with_capacity(tf.len());
        for (term, n) in tf {
            self.postings.entry(term.clone()).or_default().insert(id, n);
            distinct.push(term);
        }
        self.forward.insert(id, distinct);
        // Token counts fit u32: a metadata string holds far fewer tokens
        let len = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        self.lengths.insert(id, len);
        self.total_len += u64::from(len);
    }

    /// Remove record `id`; returns whether it was indexed
    pub(crate) fn remove(&mut self, id: u64) -> bool {
        let Some(len) = self.lengths.remove(&id) else {
            return false;
        };
        self.total_len -= u64::from(len);
        for term in self.forward.remove(&id).unwrap_or_default() {
            if let Some(ids) = self.postings.get_mut(&term) {
                ids.remove(&id);
                if ids.is_empty() {
                    self.postings.remove(&term);
                }
            }
        }
        true
    }

    /// The `k` best records for `text` by BM25, best first (ties: lower id).
    /// Only ids in `allowed` are scored when it is given.
    pub(crate) fn search(
        &self,
        text: &str,
        k: usize,
        allowed: Option<&HashSet<u64>>,
    ) -> Vec<(u64, f32)> {
        let n = self.docs() as f64;
        if n == 0.0 {
            return Vec::new();
        }
        let avgdl = self.total_len as f64 / n;
        let mut terms = tokenize(text);
        terms.sort();
        terms.dedup();
        let mut scores: HashMap<u64, f64> = HashMap::new();
        for term in &terms {
            let Some(ids) = self.postings.get(term) else {
                continue;
            };
            let df = ids.len() as f64;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for (&id, &tf) in ids {
                if allowed.is_some_and(|a| !a.contains(&id)) {
                    continue;
                }
                let tf = f64::from(tf);
                let dl = self.lengths.get(&id).copied().map_or(0.0, f64::from);
                let norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl));
                *scores.entry(id).or_insert(0.0) += idf * norm;
            }
        }
        let mut ranked: Vec<(u64, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(k);
        ranked.into_iter().map(|(id, s)| (id, s as f32)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_undoes_add() {
        let mut index = InvertedIndex::new("t");
        index.add(0, "red fox");
        let before = index.clone();
        index.add(1, "red red hen");
        assert!(index.remove(1));
        assert!(!index.remove(1));
        assert_eq!(index.postings, before.postings);
        assert_eq!(index.lengths, before.lengths);
        assert_eq!(index.total_len, before.total_len);
    }

    #[test]
    fn re_adding_replaces_the_entry() {
        let mut index = InvertedIndex::new("t");
        index.add(0, "a b");
        index.add(0, "c");
        assert_eq!(index.total_len, 1);
        assert!(!index.postings.contains_key("a"));
    }

    #[test]
    fn allowed_restricts_results() {
        let mut index = InvertedIndex::new("t");
        index.add(0, "fox");
        index.add(1, "fox");
        let allowed = HashSet::from([1]);
        let hits = index.search("fox", 10, Some(&allowed));
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [1]);
    }
}
