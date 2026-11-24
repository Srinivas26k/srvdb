---
title: db.search_batch
description: Run several searches in one call.
---

Run several searches in one call.

```python
db.search_batch(queries: Sequence[Sequence[float]], k: int) -> list[list[tuple[str, float]]]
```

## Parameters

- **queries** : *sequence of sequences of float*
  Query vectors, each as for `search`.

- **k** : *int*
  Maximum number of results, at least 1.

## Returns

`list[list[tuple[str, float]]]`: one `search` result list per query, in query order. No queries gives `[]`.

## Raises

- **ValueError**
  `k` is 0, or a query does not match the database dimension or contains NaN or infinity. Also raised for a negative value.

- **RuntimeError**
  An internal fault occurred.

## Notes

- The queries run in parallel inside Rust.
- Thread safety: releases the GIL and takes a shared lock, so calls from several threads run in parallel; they wait while a write is running. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
print(db.search_batch([v, w], k=1))
# [[('a', 0.9999999403953552)], [('b', 0.9999999403953552)]]
```

## See also

- [`db.search`](/srvdb/reference/python/db-search/): Find the k nearest vectors to a query.
