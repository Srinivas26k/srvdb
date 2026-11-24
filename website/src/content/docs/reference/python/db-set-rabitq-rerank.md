---
title: db.set_rabitq_rerank
description: Change the rerank factor of a RaBitQ database.
---

Change the rerank factor of a RaBitQ database.

```python
db.set_rabitq_rerank(rerank: int) -> None
```

## Parameters

- **rerank** : *int*
  Candidates per result rescored against the full vectors. `0` turns rerank off and returns estimated scores.

## Returns

`None`. Later searches use the new factor.

## Raises

- **ValueError**
  The database is not in RaBitQ mode, or `rerank` is above 1 000 000.

## Notes

- Higher values give higher recall and slower searches. The setting is recorded by the next `save()`.
- Thread safety: a database object is safe to share between threads. This method releases the GIL and takes the write lock, so it waits for running searches.

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128, index="rabitq", bits=1)
db.set_rabitq_rerank(80)
db.set_rabitq_rerank(0)
print("ok")
# ok
```

## See also

- [`srvdb.create`](/srvdb/reference/python/create/): Create a new database with its index and options.
