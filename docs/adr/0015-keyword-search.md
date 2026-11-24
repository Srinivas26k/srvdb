# ADR 0015: BM25 keyword search and hybrid ranking

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Retrieval-augmented applications need exact words (product codes, names,
error strings) that embeddings blur. Users pair SrvDB with SQLite FTS5 or
a second engine and fuse the results by hand (issue #79). SrvDB already
stores the text they need in each record's metadata JSON.

## Decision

1. **Source of truth.** Keyword search indexes one string field of the
   metadata JSON, chosen with `SrvDB::enable_keyword_index(field)`
   (`"a.b"` names a nested field, as in `where=` filters, ADR 0014). A
   string value is indexed; an array of strings is indexed as their
   concatenation; any other value, or a missing field, leaves the record
   out of the keyword index. The index is **derived data**: it can always
   be rebuilt from metadata and never holds the only copy of anything.

2. **Tokenizer.** Simple, deterministic and Unicode-aware, with no
   network, model or language data: the text is split into maximal runs of
   `char::is_alphanumeric` characters, each run is lowercased with Rust's
   Unicode `to_lowercase`, and runs longer than 64 characters are dropped
   (they are hashes or base64, not words). No stemming and no stop words:
   both are language-specific and would be a compatibility promise.
   Scripts written without spaces (Chinese, Japanese, Thai) become one
   token per run, so they match only whole runs; this is a known limit.

3. **Scoring.** Okapi BM25 with `k1 = 1.2`, `b = 0.75` and the
   non-negative Lucene idf `ln(1 + (N - df + 0.5) / (df + 0.5))`, where `N`
   is the number of live indexed records. Each distinct query term counts
   once. A query is at most 64 KiB of text and 1,024 distinct terms
   (`Error::InvalidInput` otherwise), checked before the index lock is
   taken, so the work of one search is bounded. Ties are broken by lower id. Deleted records are removed from the
   index (and from `N`, `df` and the average length) before scoring.

4. **Hybrid ranking.** `SrvDB::search_hybrid(query, text, k, filter)`
   takes the top `max(2k, 50)` results of vector search and of keyword
   search (both under the optional `Filter`) and fuses them with
   Reciprocal Rank Fusion, `score = sum 1 / (60 + rank)` with ranks from
   1. RRF needs no score normalisation between cosine and BM25. Ties are
   broken by lower id. The returned score is the fused score.

5. **On-disk format.** `keyword.index` in the database directory, in the
   checksummed envelope of ADR 0003/0007 (kind `KWIX`, CRC32C), replaced
   with write-to-temp, fsync, atomic rename and directory fsync. Payload
   (postcard), format 1: the field name, `covered` (records `0..covered`
   were considered), the length of every indexed record, and the postings
   `term -> [(id, term frequency)]` sorted by term and id. Load validates
   the format number, sorted unique terms and ids, ids below `covered`,
   positive frequencies and that each record's length equals the sum of
   its frequencies. Damage is `Error::ChecksumMismatch` or
   `Error::Corrupt`; calling `enable_keyword_index` again rebuilds the file
   from metadata. `enable_keyword_index` rebuilds only for those two
   errors; I/O errors are returned. `persist()` writes `keyword.index`
   after `index_settings.bin`, so a failed keyword write never leaves the
   settings unrecorded.

6. **Durability.** The index adds no guarantee of its own: what is durable
   is what `persist()` made durable in metadata (ADR 0003). `persist()`
   writes `keyword.index` after the metadata commit point, so the file can
   only lag behind metadata. On first use after open the index catches up:
   records `covered..count` are read from metadata and tokenized, and
   tombstoned records are removed once. While the index is loaded,
   `delete()` removes the record from it directly, so searches never walk
   the whole deleted set. A file claiming more records than the
   database holds is rebuilt. `compact()` renumbers ids, so before its
   commit point it atomically replaces the file with an empty one for the
   same field (`covered = 0`); a crash anywhere in compaction therefore
   leaves either the old ids with the old index or a full rebuild.

7. **API.**
   - Rust: `SrvDB::enable_keyword_index(&mut self, field)`,
     `SrvDB::keyword_field(&self)`,
     `SrvDB::keyword_search(&self, text, k, filter: Option<&Filter>)`
     (score = BM25) and `SrvDB::search_hybrid(&self, query, text, k,
     filter)`. Empty text, text with no words, `k = 0` and calling either
     search before enabling the index are `Error::InvalidInput` /
     `Error::InvalidState`.
   - Python: `db.enable_keyword_search(field="text")` and
     `db.search(query, k, *, text=None, where=None)`. With `text` and a
     query the results are fused; with `query=None` and `text` the search
     is keyword-only, so no separate `keyword_search` method is needed.
   - Keyword searches on one database are serialized by a mutex around
     the in-memory index (it may catch up during a search); vector searches
     are unaffected.

## Alternatives considered

- **Tantivy or SQLite FTS5 inside SrvDB.** Mature, but a large dependency
  (or a C library) for one field and a second storage engine with its own
  durability rules. Revisit if the benchmark shows BM25 here is too slow.
- **Weighted score fusion.** Needs per-dataset normalisation of cosine and
  BM25 scores; RRF is parameter-light and robust.
- **Indexing during `add`.** Couples every write path to the tokenizer;
  catching up from metadata keeps the write path unchanged and makes the
  rebuild path the normal path, so it is always exercised.
- **Stemming / stop words.** Language-specific; can be added later as an
  opt-in analyser with a format bump.

## Consequences

- New public API (above) and a new owned file, `keyword.index`.
- The first keyword search after opening a database whose index lags
  reads the missing records' metadata; a full rebuild is O(n).
- Speed and quality are compared with SQLite FTS5 by
  `scripts/bench_keyword.py`; no numbers are published from it until run
  on a quiet machine with the details required by AGENTS.md section 11.
