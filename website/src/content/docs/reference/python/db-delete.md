---
title: db.delete
description: Delete vectors by id.
---

Delete vectors by id.

```python
db.delete(ids: Sequence[str]) -> int
```

## Parameters

- **ids** : *sequence of str*
  Ids to delete. Unknown ids are ignored.

## Returns

`int`: the number of vectors that existed and were deleted.

## Raises

- **OSError**
  Writing the files failed.

- **RuntimeError**
  An internal fault occurred.

## Notes

- Deleted vectors disappear from `search`, `count` and `get_all_ids` immediately; the disk space is reclaimed by [`compact`](/srvdb/reference/python/db-compact/).
- Durability: the write is visible to `search` as soon as the call returns, but it is durable only after a later [`save()`](/srvdb/reference/python/db-save/) returns. After a crash the database reopens in the state of the last successful `save()`; see [Durability and save()](/srvdb/guides/durability/).
- Thread safety: a database object is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `save`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
print(db.delete(["a", "missing"]))
# 1
db.save()
print(db.count())
# 1
```

## See also

- [`db.compact`](/srvdb/reference/python/db-compact/): Remove deleted and replaced vectors from disk.
- [`db.upsert`](/srvdb/reference/python/db-upsert/): Add vectors, replacing any that already have the same id.
- [`db.save`](/srvdb/reference/python/db-save/): Make every earlier write durable (crash-safe commit).
