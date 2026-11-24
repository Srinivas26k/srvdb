# ADR 0014: Declarative metadata filters (`where=`) in the core and Python

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

ADR 0010 added filtered search to the Rust core as an id set
(`search_filtered`) built from a Rust closure over metadata JSON
(`ids_where`). Python had no way to use it: a Python callable cannot run
inside the core with the GIL released, and handing Python a set of
internal `u64` ids would leak an internal numbering that changes on
`compact()`. The LangChain and LlamaIndex adapters (ADR 0009) post-filtered
the search results, which returns short or costs close to a full scan for
selective filters (issue #46).

## Decision

1. **A declarative filter type in the core.** `srvdb::Filter::from_json`
   parses a JSON object into a validated tree; `Filter::matches(&str)`
   tests one metadata record. The grammar is a fixed set, like the subset
   other embedded stores (Chroma, MongoDB-style) use:
   - `{"field": value}` equality (string, number, bool or null; numbers
     compare by value so `1 == 1.0`);
   - `{"field": {"$eq"|"$ne"|"$gt"|"$gte"|"$lt"|"$lte"|"$in"|"$nin": v}}`;
   - `$and`, `$or` (non-empty lists) and `$not` at any object level;
   - several keys in one object are ANDed; `"a.b"` is a nested field.
   A missing field matches only `$ne` and `$nin`. Ordering compares
   numbers with numbers and strings with strings; other pairs never match.
   Anything else (empty objects, unknown operators, a list as an equality
   value, a bool for `$gt`) is `Error::InvalidInput`, so Rust and Python
   users get the same validation.
2. **Two engine methods.** `SrvDB::search_where(query, k, &Filter)` and
   `SrvDB::search_batch_where(queries, k, &Filter)` build the id set with
   `ids_where` and call `search_filtered`, keeping every guarantee of
   ADR 0010 (k results when k match, exact scan for selective filters).
   The batch form reads the metadata once for all queries.
3. **Python: a keyword-only `where=` dict.**
   `db.search(query, k, *, where=None)` and
   `db.search_batch(queries, k, *, where=None)`. The binding only converts
   the dict to JSON (`json.dumps`, `allow_nan=False`) and maps errors:
   every invalid filter is `ValueError`. The GIL is released for the whole
   search, including the metadata pass.
4. **Adapters use it.** LangChain dict filters (equality or the operators
   above) and LlamaIndex `MetadataFilters`, `node_ids` and `doc_ids` are
   translated to `where`, prefixing fields with the adapter's `metadata.`
   record key. Python callables (LangChain) and LlamaIndex keys containing
   a dot (which `where` reads as nesting) still post-filter.

## Alternatives considered

- **Expose the id set to Python** (`db.ids_where(...)` returning ids).
  Leaks internal ids, which `compact()` renumbers, and still needs a
  filter language or a Python callback per record.
- **A Python callable evaluated per record.** Holds the GIL for an O(n)
  pass and blocks parallel searches; it is still what the LangChain
  adapter does for callables.
- **Caching the id set per filter.** Saves the O(n) metadata pass per
  call but needs invalidation on every write. Not done until a benchmark
  shows the pass dominates; `search_batch(where=)` already shares it.
- **Regex, `$contains` or full-text operators.** Not needed by the
  adapters' supported operators; each would be a further compatibility
  promise.

## Consequences

- `srvdb::Filter`, `SrvDB::search_where` and `SrvDB::search_batch_where`
  are new public API; the Python `search` / `search_batch` gain a
  keyword-only argument (backwards compatible).
- A filtered call costs one pass over the metadata store plus the
  filtered search. No on-disk format changes.
- LlamaIndex `NIN` on a missing key now matches (it did not before), so
  `NIN` is the negation of `IN` in both the core and the Python fallback.
- LangChain dict filters whose values are dicts are now operator objects,
  not equality with a dict; list values are rejected (use `$in`).
