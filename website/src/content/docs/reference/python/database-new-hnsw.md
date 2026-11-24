---
title: srvdb.Database.new_hnsw
description: Open or create an HNSW database with explicit graph parameters.
---

Open or create an HNSW database with explicit graph parameters.

```python
srvdb.Database.new_hnsw(path: str, dimension: int = 1536, m: int = 16, ef_construction: int = 200, ef_search: int = 50) -> srvdb.Database
```

## Parameters

- **path** : *str*
  Directory holding the database files.

- **dimension** : *int*, default `1536`
  Vector dimension, 128 to 4096.

- **m** : *int*, default `16`
  Neighbours per graph node. Higher uses more memory and gives better recall.

- **ef_construction** : *int*, default `200`
  Candidate list size while building the graph.

- **ef_search** : *int*, default `50`
  Candidate list size while searching; change it later with [`set_ef_search`](/srvdb/reference/python/database-set-ef-search/).

## Returns

`Database` in `"hnsw"` mode.

## Raises

- **ValueError**
  `dimension` is outside 128 to 4096.

- **BlockingIOError**
  The database is already open elsewhere.

- **OSError**
  The files cannot be created or read.

- **RuntimeError**
  Stored data is damaged or an internal fault occurred.

## Notes

- `m`, `ef_construction` and `ef_search` are not range-checked here (see the open points in the issue tracker); use positive values.
- The HNSW index is derived data: it is rebuilt from the stored vectors if its file is missing or damaged.
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.Database.new_hnsw("./vectors", dimension=128, m=8, ef_search=64)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a"], [v], ["{}"])
print(db.search(v, k=1))
# [('a', 0.9999999403953552)]
```

## See also

- [`srvdb.open`](/srvdb/reference/python/open/): Open a database, creating it if it does not exist.
- [`srvdb.Database.set_ef_search`](/srvdb/reference/python/database-set-ef-search/): Set HNSW's search candidate list size.
- [`srvdb.Database.set_mode`](/srvdb/reference/python/database-set-mode/): Switch the index mode of an open database.
