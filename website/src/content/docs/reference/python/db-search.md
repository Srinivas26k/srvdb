---
title: db.search
description: Find the k nearest vectors to a query.
---

Find the k nearest vectors to a query.

```python
db.search(query: Sequence[float], k: int) -> list[tuple[str, float]]
```

## Parameters

- **query** : *sequence of float*
  Query vector with the database dimension and only finite values (a list or 1-D NumPy array).

- **k** : *int*
  Maximum number of results, at least 1.

## Returns

`list[tuple[str, float]]`: up to `k` `(id, score)` pairs, best first. Scores are cosine similarities computed in float32, so an identical vector scores about 0.99999994 rather than exactly 1.0. Fewer than `k` pairs come back when the database holds fewer vectors; an empty database returns `[]`.

## Raises

- **ValueError**
  `k` is 0, or a query does not match the database dimension or contains NaN or infinity. Also raised for a negative value.

- **RuntimeError**
  An internal fault occurred.

## Notes

- Sees every write whose call has returned, persisted or not.
- Results are exact in `"flat"` mode and approximate in the other modes. In `"pq"` mode scores are approximate too.
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
```

## See also

- [`db.search_batch`](/srvdb/reference/python/db-search-batch/): Run several searches in one call.
- [`db.get`](/srvdb/reference/python/db-get/): Return the metadata stored for an id.
- [`db.set_ef_search`](/srvdb/reference/python/db-set-ef-search/): Set HNSW's search candidate list size.
