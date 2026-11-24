---
title: srvdb.Database.count
description: Return the number of searchable vectors.
---

Return the number of searchable vectors.

```python
Database.count() -> int
```

## Parameters

None.

## Returns

`int`: the number of vectors that are not deleted.

## Raises

Nothing in normal use.

## Notes

- Never raises for valid use; a `RuntimeError` is possible only after an earlier internal fault.
- Thread safety: releases the GIL and takes a shared lock, so calls from several threads run in parallel; they wait while a write is running. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.open("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
db.delete(["a"])
print(db.count())
# 1
```

## See also

- [`srvdb.Database.get_all_ids`](/srvdb/reference/python/database-get-all-ids/): Return every live id.
- [`srvdb.Database.info`](/srvdb/reference/python/database-info/): Return a one-line summary of the database.
