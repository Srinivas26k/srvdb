---
title: db.info
description: Return a one-line summary of the database.
---

Return a one-line summary of the database.

```python
db.info() -> str
```

## Parameters

None.

## Returns

`str`: dimension, mode and vector count, plus memory and compression in `"pq"` mode. `repr(db)` returns the same text.

## Raises

Nothing in normal use.

## Notes

- The format is for people, not for parsing; it may change.
- Thread safety: releases the GIL and takes a shared lock, so calls from several threads run in parallel; they wait while a write is running. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
print(db.info())
# SrvDB(dimension=128, mode='hnsw', vectors=0)
```

## See also

- [`db.count`](/srvdb/reference/python/db-count/): Return the number of searchable vectors.
