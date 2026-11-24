---
title: srvdb.Database.upsert
description: Add vectors, replacing any that already have the same id.
---

Add vectors, replacing any that already have the same id.

```python
Database.upsert(ids: Sequence[str], embeddings: Sequence[Sequence[float]], metadatas: Sequence[str]) -> int
```

## Parameters

- **ids** : *sequence of str*
  Non-empty string ids, unique within the batch.

- **embeddings** : *sequence of sequences of float*
  One vector per id, each with the database dimension and only finite values. Lists, tuples and 1-D NumPy arrays work; values are copied to `float32`.

- **metadatas** : *sequence of str*
  One JSON object string per id, for example `json.dumps({"title": "A"})`. Use `"{}"` for none.

## Returns

`int`: the number of vectors written.

## Raises

- **ValueError**
  As for [`add`](/srvdb/reference/python/database-add/), except that existing ids are allowed.

- **OSError**
  Writing the files failed.

- **RuntimeError**
  An internal fault occurred.

## Notes

- An existing id afterwards refers to its new vector and metadata; the old vector is marked deleted and its space is reclaimed by [`compact`](/srvdb/reference/python/database-compact/).
- Durability: the write is visible to `search` as soon as the call returns, but it is durable only after a later [`persist()`](/srvdb/reference/python/database-persist/) returns. After a crash the database reopens in the state of the last successful `persist()`; see [Durability and persist()](/srvdb/guides/durability/).
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.open("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a"], [v], ["{}"])
db.upsert(["a"], [w], ['{"version": 2}'])
print(db.count(), db.search(w, k=1))
# 1 [('a', 0.9999999403953552)]
db.persist()
```

## See also

- [`srvdb.Database.add`](/srvdb/reference/python/database-add/): Add new vectors with their ids and metadata.
- [`srvdb.Database.delete`](/srvdb/reference/python/database-delete/): Delete vectors by id.
- [`srvdb.Database.compact`](/srvdb/reference/python/database-compact/): Remove deleted and replaced vectors from disk.
