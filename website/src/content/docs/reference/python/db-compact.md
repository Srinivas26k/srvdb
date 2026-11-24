---
title: db.compact
description: Remove deleted and replaced vectors from disk.
---

Remove deleted and replaced vectors from disk.

```python
db.compact() -> int
```

## Parameters

None.

## Returns

`int`: the number of vectors removed.

## Raises

- **OSError**
  Writing the files failed.

- **RuntimeError**
  An internal fault occurred.

## Notes

- Persists first, so everything written before the call is durable when it returns.
- String ids are unchanged, and search results are the same before and after. HNSW and IVF indexes are rebuilt, which takes time proportional to the data.
- Thread safety: a database object is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `save`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
db.delete(["a"])
print(db.compact())
# 1
```

## See also

- [`db.delete`](/srvdb/reference/python/db-delete/): Delete vectors by id.
- [`db.upsert`](/srvdb/reference/python/db-upsert/): Add vectors, replacing any that already have the same id.
- [`db.save`](/srvdb/reference/python/db-save/): Make every earlier write durable (crash-safe commit).
