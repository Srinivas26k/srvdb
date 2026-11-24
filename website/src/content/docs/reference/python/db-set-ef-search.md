---
title: srvdb.Database.set_ef_search
description: Set HNSW's search candidate list size.
---

Set HNSW's search candidate list size.

```python
Database.set_ef_search(ef_search: int) -> None
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

- Applies to the HNSW index and to every IVF partition, and is saved with the index by `persist()`.
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.Database.new_hnsw("./vectors", dimension=128)
db.set_ef_search(100)
```

## See also

- [`srvdb.Database.new_hnsw`](/srvdb/reference/python/database-new-hnsw/): Open or create an HNSW database with explicit graph parameters.
- [`srvdb.Database.search`](/srvdb/reference/python/database-search/): Find the k nearest vectors to a query.
