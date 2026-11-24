---
title: db.set_ef_search
description: Set HNSW's search candidate list size.
---

Set HNSW's search candidate list size.

```python
db.set_ef_search(ef_search: int) -> None
```

## Parameters

- **ef_search** : *int*
  Candidate list size, at least 1. Higher finds more true neighbours but searches more slowly; typical values are 50 to 300. Searches always use at least `k`.

## Returns

`None`.

## Raises

- **ValueError**
  `ef_search` is 0.

- **OverflowError**
  `ef_search` is negative.

## Notes

- Applies to the HNSW index and to every IVF partition, and is saved with the index by `save()`.
- Thread safety: a database object is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `save`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128, index="hnsw")
db.set_ef_search(100)
```

## See also

- [`srvdb.create`](/srvdb/reference/python/create/): Create a new database with its index and options.
- [`db.search`](/srvdb/reference/python/db-search/): Find the k nearest vectors to a query.
