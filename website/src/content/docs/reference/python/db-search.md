---
title: db.search
description: Find the k nearest vectors to a query.
---

Find the k nearest vectors to a query.

```python
db.search(query: Sequence[float] | None, k: int, *, text: str | None = None, where: dict | None = None) -> list[tuple[str, float]]
```

## Parameters

- **query** : *sequence of float or None*
  Query vector with the database dimension and only finite values (a list or 1-D NumPy array), or `None` for a keyword-only search with `text`.

- **k** : *int*
  Maximum number of results, at least 1.

- **text** : *str, optional, keyword-only*
  Keyword query, ranked by BM25 over the field chosen with [`db.enable_keyword_search`](/srvdb/reference/python/db-enable-keyword-search/). With a `query` vector the two rankings are fused (Reciprocal Rank Fusion). See [Keyword and hybrid search](/srvdb/guides/keyword-search/).

- **where** : *dict, optional, keyword-only*
  Metadata filter, for example `{"genre": "jazz"}` or `{"year": {"$gte": 1950}}`. Operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`, `$not`; `"a.b"` names a nested field. See [Filtered search](/srvdb/guides/filtered-search/). Default `None` searches every vector.

## Returns

`list[tuple[str, float]]`: up to `k` `(id, score)` pairs, best first. Scores are cosine similarities computed in float32, so an identical vector scores about 0.99999994 rather than exactly 1.0. With `text` and no query, scores are BM25 scores; with both, they are fused rank scores that only order the results. With `where`, only matching vectors are returned, and `k` of them whenever at least `k` match. Fewer than `k` pairs come back when fewer vectors (or matches) exist; an empty database returns `[]`.

## Raises

- **ValueError**
  `k` is 0, or a query does not match the database dimension or contains NaN or infinity, or `where` is not a valid filter (empty, an unknown operator, a wrong operand type, a value JSON cannot encode). Also raised for a negative `k`, when both `query` and `text` are `None`, when `text` has no letters or digits, and when `text` is given before `enable_keyword_search()`.

- **RuntimeError**
  An internal fault occurred, or the keyword index file is damaged (call `enable_keyword_search()` to rebuild it).

## Notes

- Sees every write whose call has returned, persisted or not.
- Results are exact in `"flat"` mode and approximate in the other modes. In `"pq"` mode scores are approximate too.
- A `where` filter reads every metadata record once per call (O(n)); `search_batch` shares that pass across queries.
- Thread safety: releases the GIL and takes a shared lock, so calls from several threads run in parallel; they wait while a write is running. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
print(db.search(v, k=2))
# [('a', 0.9999999403953552), ('b', 0.0)]

db.add(["c"], [w], ['{"kind": "note"}'])
print(db.search(v, k=2, where={"kind": "note"}))
# [('c', 0.0)]
```

## See also

- [Filtered search](/srvdb/guides/filtered-search/): Filter syntax and performance.
- [`db.search_batch`](/srvdb/reference/python/db-search-batch/): Run several searches in one call.
- [Keyword and hybrid search](/srvdb/guides/keyword-search/): BM25 and fusion.
- [`db.get`](/srvdb/reference/python/db-get/): Return the metadata stored for an id.
- [`db.set_ef_search`](/srvdb/reference/python/db-set-ef-search/): Set HNSW's search candidate list size.
