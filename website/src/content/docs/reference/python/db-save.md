---
title: db.save
description: Make every earlier write durable (crash-safe commit).
---

Make every earlier write durable (crash-safe commit).

```python
db.save() -> None
db.persist() -> None  # alias
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

- Guarantee: when `save()` returns, every earlier `add`, `upsert` and `delete` survives a process crash or power loss. After a crash the database reopens in exactly the state of the last successful `save()`; later writes are absent, never partially present. See [Durability and save()](/srvdb/guides/durability/).
- Writes not followed by `save()` are lost when the process exits or the object is garbage-collected.
- `persist()` is the same method under its 0.3 name.
- A `with` block calls `save()` when it ends without an exception; see [the database object](/srvdb/reference/python/db/).
- `save()` fsyncs, so it costs real time: call it per batch or at checkpoints, not per vector.
- Thread safety: a database object is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `save`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a"], [v], ["{}"])
db.save()
db.close()  # releases the lock
print(srvdb.open("./vectors").count())
# 1
```

## See also

- [`db.add`](/srvdb/reference/python/db-add/): Add new vectors with their ids and metadata.
- [`db.compact`](/srvdb/reference/python/db-compact/): Remove deleted and replaced vectors from disk.
