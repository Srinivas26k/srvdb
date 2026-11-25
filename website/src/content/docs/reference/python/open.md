---
title: srvdb.open
description: Open an existing database by path.
---

Open an existing database. The dimension, index and options are read from disk, so the path is all you pass.

```python
srvdb.open(path: str) -> db
```

## Parameters

- **path** : *str*
  Directory of a database made by [`srvdb.create`](/srvdb/reference/python/create/). Used exactly as given; nothing is created.

## Returns

The open database object (`db`). Use it in a `with` block to save and close it on exit; see [the database object](/srvdb/reference/python/db/).

## Raises

- **FileNotFoundError**
  No database exists at `path`.

- **ValueError**
  A quantized database written by SrvDB 0.3 that has no recorded settings yet; see [Upgrading](/srvdb/guides/upgrading/).

- **BlockingIOError**
  The database is already open, in this process or another (an `OSError` subclass).

- **OSError**
  The files cannot be read.

- **RuntimeError**
  Stored data is damaged (checksum mismatch) or an internal fault occurred.

## Notes

- It shadows the builtin `open` only inside the `srvdb` namespace; call it as `srvdb.open`.
- Databases written by SrvDB 0.3 with full-precision vectors open as flat databases, as before.
- One database object holds a lock on the directory until [`db.close()`](/srvdb/reference/python/db-close/), the end of a `with` block, or garbage collection; opening the same directory again before then raises `BlockingIOError`.
- Thread safety: the returned object is safe to share between threads. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

srvdb.create("./vectors", dimension=128, index="flat").close()
with srvdb.open("./vectors") as db:
    print(db)
# SrvDB(dimension=128, mode='flat', vectors=0)
```

```python
import srvdb

try:
    srvdb.open("./missing")
except FileNotFoundError:
    print("no database yet")
# no database yet
```

## See also

- [`srvdb.create`](/srvdb/reference/python/create/): Create a new database with its index and options.
- [The database object](/srvdb/reference/python/db/): What `open` returns.
- [`db.save`](/srvdb/reference/python/db-save/): Make every earlier write durable (crash-safe commit).
