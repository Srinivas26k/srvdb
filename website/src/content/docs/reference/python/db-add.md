---
title: db.add
description: Add new vectors with their ids and metadata.
---

Add new vectors with their ids and metadata.

```python
db.add(ids: Sequence[str], embeddings: Sequence[Sequence[float]], metadatas: Sequence[str]) -> int
```

## Parameters

- **ids** : *sequence of str*
  Non-empty string ids, unique within the batch. None may exist in the database already.

- **embeddings** : *sequence of sequences of float*
  One vector per id, each with the database dimension and only finite values. Lists, tuples and 1-D NumPy arrays work; values are copied to `float32`.

- **metadatas** : *sequence of str*
  One JSON object string per id, for example `json.dumps({"title": "A"})`. Use `"{}"` for none.

## Returns

`int`: the number of vectors added.

## Raises

- **ValueError**
  `ids`, `embeddings` and `metadatas` differ in length; an id is empty or repeated within the batch; a vector does not match the database dimension or contains NaN or infinity; or a metadata string is not valid JSON or not a JSON object. Nothing is written. Also raised when an id already exists; use [`upsert`](/srvdb/reference/python/db-upsert/) to replace it.

- **OSError**
  Writing the files failed.

- **RuntimeError**
  An internal fault occurred.

## Notes

- Durability: the write is visible to `search` as soon as the call returns, but it is durable only after a later [`save()`](/srvdb/reference/python/db-save/) returns. After a crash the database reopens in the state of the last successful `save()`; see [Durability and save()](/srvdb/guides/durability/).
- Thread safety: a database object is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `save`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import json
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
n = db.add(["a", "b"], [v, w], [json.dumps({"title": "A"}), "{}"])
print(n)
# 2
db.save()
```

## See also

- [`db.upsert`](/srvdb/reference/python/db-upsert/): Add vectors, replacing any that already have the same id.
- [`db.save`](/srvdb/reference/python/db-save/): Make every earlier write durable (crash-safe commit).
- [`db.delete`](/srvdb/reference/python/db-delete/): Delete vectors by id.
