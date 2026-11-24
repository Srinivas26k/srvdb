---
title: The database object (db)
description: What srvdb.create and srvdb.open return, and how to use it as a context manager.
---

[`srvdb.create`](/srvdb/reference/python/create/) and [`srvdb.open`](/srvdb/reference/python/open/) return an open database object, called `db` throughout these pages. You never construct it directly.

```python
with srvdb.open(path) as db:   # saves and closes on a clean exit
    db.add(ids, vectors, metadatas)
    db.search(query, k=10)
```

## Context manager

- A `with` block that ends normally calls [`db.save()`](/srvdb/reference/python/db-save/) and then [`db.close()`](/srvdb/reference/python/db-close/).
- A `with` block that ends with an exception calls only `db.close()`: writes since the last `save()` are **not** made durable, so a half-finished batch never reaches disk. The exception propagates.
- After the block, every method except `close()` raises `ValueError`.

## Methods

| Method | Summary |
|--------|---------|
| [`db.add`](/srvdb/reference/python/db-add/) | Add new vectors with their ids and metadata. |
| [`db.upsert`](/srvdb/reference/python/db-upsert/) | Add vectors, replacing any that already have the same id. |
| [`db.delete`](/srvdb/reference/python/db-delete/) | Delete vectors by id. |
| [`db.save`](/srvdb/reference/python/db-save/) | Make every earlier write durable. `db.persist()` is an alias. |
| [`db.close`](/srvdb/reference/python/db-close/) | Release the database and its lock without saving. |
| [`db.search`](/srvdb/reference/python/db-search/) | Find the k nearest vectors to a query. |
| [`db.count`](/srvdb/reference/python/db-count/) | Number of searchable vectors. |

The full list is in the [Python API overview](/srvdb/reference/python/).

## Notes

- Vectors are identified by string ids. Each vector carries one metadata JSON object; SrvDB stores the id in it under the key `"__id__"`.
- Scores are cosine similarities, higher is closer.
- Durability: writes are visible to `search` as soon as the call returns and durable once a later `save()` returns. See [Durability and save()](/srvdb/guides/durability/).
- Thread safety: a database object is safe to share between threads. Every method releases the GIL; writes take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).
- For type hints, the class is importable as `from srvdb import Database`.

## Examples

```python
import srvdb

v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
with srvdb.create("./vectors", dimension=128) as db:
    db.add(["a", "b"], [v, w], ['{"title": "A"}', "{}"])
with srvdb.open("./vectors") as db:
    print(db.search(v, k=1))
# [('a', 1.0)]
```

```python
import srvdb

with srvdb.create("./vectors", dimension=128) as db:
    db.add(["a"], [[1.0] + [0.0] * 127], ["{}"])
try:
    with srvdb.open("./vectors") as db:
        db.add(["b"], [[0.0, 1.0] + [0.0] * 126], ["{}"])
        raise RuntimeError("embedding service failed")
except RuntimeError:
    pass
print(srvdb.open("./vectors").count())  # "b" was not saved
# 1
```
