---
title: srvdb.open
description: Open a database, creating it if it does not exist.
---

Open a database, creating it if it does not exist.

```python
srvdb.open(path: str, dimension: int = 1536, mode: str = "flat") -> srvdb.Database
```

## Parameters

- **path** : *str*
  Directory holding the database files. Used exactly as given.

- **dimension** : *int*, default `1536`
  Vector dimension, 128 to 4096. Must match the stored dimension when the database exists.

- **mode** : *str*, default `"flat"`
  Index mode, case-insensitive: `"flat"`, `"hnsw"`, `"sq8"` (alias `"scalar"`), `"pq"` (alias `"product"`), `"ivf"` or `"auto"`. See [Choosing an index mode](/srvdb/guides/index-modes/).

## Returns

`Database`: the open database.

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

- Same as `srvdb.Database(path, dimension=dimension, mode=mode)`. It shadows the builtin `open` only inside the `srvdb` namespace; call it as `srvdb.open`.
- The mode is not stored on disk: pass it every time you open the database.
- One `Database` object holds a lock on the directory until it is garbage-collected; opening the same directory again before then raises `BlockingIOError`.
- In `"pq"` mode the codebooks are trained on the first batch added, which must hold at least 256 vectors; otherwise `add` raises `ValueError`. Use [`Database.new_product_quantized`](/srvdb/reference/python/database-new-product-quantized/) to train on a sample you choose.
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.open("./vectors", dimension=128)
print(db)
# SrvDB(dimension=128, mode='flat', vectors=0)
```

## See also

- [`srvdb.Database`](/srvdb/reference/python/database/): An open SrvDB database.
- [`srvdb.Database.new_hnsw`](/srvdb/reference/python/database-new-hnsw/): Open or create an HNSW database with explicit graph parameters.
- [`srvdb.Database.persist`](/srvdb/reference/python/database-persist/): Make every earlier write durable (crash-safe commit).
