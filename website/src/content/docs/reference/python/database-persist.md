---
title: srvdb.Database.persist
description: Make every earlier write durable (crash-safe commit).
---

Make every earlier write durable (crash-safe commit).

```python
Database.persist() -> None
```

## Parameters

None.

## Returns

`None`.

## Raises

- **OSError**
  Writing or syncing the files failed.

- **RuntimeError**
  An internal fault occurred.

## Notes

- Guarantee: when `persist()` returns, every earlier `add`, `upsert` and `delete` survives a process crash or power loss. After a crash the database reopens in exactly the state of the last successful `persist()`; later writes are absent, never partially present. See [Durability and persist()](/srvdb/guides/durability/).
- Writes not followed by `persist()` are lost when the process exits or the object is garbage-collected.
- `persist()` fsyncs, so it costs real time: call it per batch or at checkpoints, not per vector.
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.open("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a"], [v], ["{}"])
db.persist()
del db  # releases the lock
print(srvdb.open("./vectors", dimension=128).count())
# 1
```

## See also

- [`srvdb.Database.add`](/srvdb/reference/python/database-add/): Add new vectors with their ids and metadata.
- [`srvdb.Database.compact`](/srvdb/reference/python/database-compact/): Remove deleted and replaced vectors from disk.
