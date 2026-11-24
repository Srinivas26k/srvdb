//! `keyword.index`: the on-disk inverted index (ADR 0015)
//!
//! Checksummed envelope (kind `KWIX`, CRC32C), replaced atomically. The
//! postcard payload has its own `format` number. Everything is validated
//! on load; the file is derived data and can always be rebuilt.

use super::index::InvertedIndex;
use crate::error::Error;
use crate::storage::{atomic::write_atomic, format};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// File name inside the database directory
pub(crate) const FILE_NAME: &str = "keyword.index";
/// Payload format written by this version
const FORMAT: u16 = 1;

/// On-disk payload
#[derive(Debug, Serialize, Deserialize)]
struct Record {
    format: u16,
    field: String,
    covered: u64,
    /// (id, token count), sorted by id
    lengths: Vec<(u64, u32)>,
    /// (term, [(id, term frequency)]), sorted by term, then id
    postings: Vec<(String, Vec<(u64, u32)>)>,
}

/// Write `index` atomically to `dir`
pub(crate) fn save(dir: &Path, index: &InvertedIndex) -> Result<()> {
    let record = Record {
        format: FORMAT,
        field: index.field.clone(),
        covered: index.covered,
        lengths: index.lengths.iter().map(|(&id, &n)| (id, n)).collect(),
        postings: index
            .postings
            .iter()
            .map(|(t, ids)| (t.clone(), ids.iter().map(|(&id, &n)| (id, n)).collect()))
            .collect(),
    };
    write_atomic(
        &dir.join(FILE_NAME),
        &format::encode(format::KEYWORD_INDEX, &record)?,
    )
}

/// Read the index from `dir`, or `None` if there is no file
pub(crate) fn load(dir: &Path) -> Result<Option<InvertedIndex>> {
    let path = dir.join(FILE_NAME);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("failed to read {}", path.display())),
    };
    let record: Record = format::decode(format::KEYWORD_INDEX, &bytes)?;
    from_record(record).map(Some)
}

fn corrupt(why: &str) -> anyhow::Error {
    Error::Corrupt(format!(
        "{FILE_NAME}: {why}; call enable_keyword_index again to rebuild it from metadata"
    ))
    .into()
}

fn sorted_unique<T: Ord>(items: impl Iterator<Item = T>) -> bool {
    let mut prev: Option<T> = None;
    for item in items {
        if prev.as_ref().is_some_and(|p| *p >= item) {
            return false;
        }
        prev = Some(item);
    }
    true
}

fn from_record(r: Record) -> Result<InvertedIndex> {
    if r.format != FORMAT {
        return Err(corrupt(&format!(
            "payload format {} is not supported (expected {FORMAT})",
            r.format
        )));
    }
    if r.field.is_empty() {
        return Err(corrupt("the field name is empty"));
    }
    if !sorted_unique(r.lengths.iter().map(|l| l.0))
        || !sorted_unique(r.postings.iter().map(|p| &p.0))
    {
        return Err(corrupt("records or terms are not sorted and unique"));
    }
    let lengths: BTreeMap<u64, u32> = r.lengths.into_iter().collect();
    if lengths
        .keys()
        .next_back()
        .is_some_and(|&id| id >= r.covered)
    {
        return Err(corrupt("a record id is beyond the covered range"));
    }
    let mut sums: HashMap<u64, u64> = HashMap::new();
    let mut forward: HashMap<u64, Vec<String>> = HashMap::new();
    let mut postings = BTreeMap::new();
    for (term, ids) in r.postings {
        let valid = !ids.is_empty()
            && sorted_unique(ids.iter().map(|p| p.0))
            && ids.iter().all(|(id, n)| *n > 0 && lengths.contains_key(id));
        if !valid {
            return Err(corrupt(
                "a posting list is empty, unsorted or names an unknown record",
            ));
        }
        for &(id, n) in &ids {
            *sums.entry(id).or_insert(0) += u64::from(n);
            forward.entry(id).or_default().push(term.clone());
        }
        postings.insert(term, ids.into_iter().collect());
    }
    if lengths
        .iter()
        .any(|(id, &n)| sums.get(id).copied() != Some(u64::from(n)))
    {
        return Err(corrupt(
            "a record length does not match its term frequencies",
        ));
    }
    let total_len = lengths.values().map(|&n| u64::from(n)).sum();
    Ok(InvertedIndex {
        field: r.field,
        covered: r.covered,
        postings,
        lengths,
        total_len,
        forward,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trips() {
        let dir = TempDir::new().unwrap();
        let mut index = InvertedIndex::new("body");
        index.add(0, "the quick fox");
        index.add(2, "lazy fox fox");
        index.covered = 3;
        save(dir.path(), &index).unwrap();
        let back = load(dir.path()).unwrap().unwrap();
        assert_eq!(back.field, "body");
        assert_eq!(back.covered, 3);
        assert_eq!(back.postings, index.postings);
        assert_eq!(back.lengths, index.lengths);
        assert_eq!(back.total_len, index.total_len);
        assert_eq!(back.search("fox", 5, None), index.search("fox", 5, None));
    }

    #[test]
    fn missing_file_is_none() {
        let dir = TempDir::new().unwrap();
        assert!(load(dir.path()).unwrap().is_none());
    }

    #[test]
    fn inconsistent_payload_is_corrupt() {
        let dir = TempDir::new().unwrap();
        let bad = Record {
            format: FORMAT,
            field: "t".into(),
            covered: 1,
            lengths: vec![(0, 5)],
            postings: vec![("a".into(), vec![(0, 1)])],
        };
        let bytes = format::encode(format::KEYWORD_INDEX, &bad).unwrap();
        std::fs::write(dir.path().join(FILE_NAME), bytes).unwrap();
        let err = Error::from(load(dir.path()).unwrap_err());
        assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
    }
}
