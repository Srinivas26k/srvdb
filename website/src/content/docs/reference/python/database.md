---
title: srvdb.Database
description: An open SrvDB database.
---

An open SrvDB database.

```python
class srvdb.Database(path: str, dimension: int = 1536, mode: str = "flat")
```

## Parameters

- **path** : *str*
  Directory holding the database files. Used exactly as given.

- **dimension** : *int*, default `1536`
  Vector dimension, 128 to 4096. Must match the stored dimension when the database exists.

- **mode** : *str*, default `"flat"`
  Index mode, case-insensitive: `"flat"`, `"hnsw"`, `"sq8"` (alias `"scalar"`), `"pq"` (alias `"product"`), `"ivf"` or `"auto"`. Recorded on disk: the default `"flat"` reopens in the recorded mode, another mode raises `ValueError`.

## Raises

- **ValueError**
  `dimension` is outside 128 to 4096, `mode` is unknown, or `dimension` does not match an existing database.

- **BlockingIOError**
  The database is already open, in this process or another (an `OSError` subclass).

- **OSError**
  The files cannot be created or read.

- **RuntimeError**
  Stored data is damaged (checksum mismatch) or an internal fault occurred.

## Notes

- Prefer [`srvdb.open`](/srvdb/reference/python/open/), which is the same call. The class stays public for type hints and the alternative constructors.
- Vectors are identified by string ids. Each vector carries one metadata JSON object; SrvDB stores the id in it under the key `"__id__"`.
- Scores are cosine similarities, higher is closer.
- Durability: the write is visible to `search` as soon as the call returns, but it is durable only after a later [`persist()`](/srvdb/reference/python/database-persist/) returns. After a crash the database reopens in the state of the last successful `persist()`; see [Durability and persist()](/srvdb/guides/durability/).
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.Database("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ['{"title": "A"}', "{}"])
db.persist()
print(db.search(v, k=1))
# [('a', 0.9999999403953552)]
```

## See also

- [`srvdb.open`](/srvdb/reference/python/open/): Open a database, creating it if it does not exist.
- [`srvdb.Database.add`](/srvdb/reference/python/database-add/): Add new vectors with their ids and metadata.
- [`srvdb.Database.search`](/srvdb/reference/python/database-search/): Find the k nearest vectors to a query.
