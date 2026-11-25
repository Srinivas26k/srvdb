---
title: db.get_all_ids
description: Return every live id.
---

Return every live id.

```python
db.get_all_ids() -> list[str]
```

## Parameters

None.

## Returns

`list[str]`: every id that is not deleted, in no particular order.

## Raises

Nothing in normal use.

## Notes

- Never raises for valid use; a `RuntimeError` is possible only after an earlier internal fault.
- Thread safety: releases the GIL and takes a shared lock, so calls from several threads run in parallel; they wait while a write is running. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
print(sorted(db.get_all_ids()))
# ['a', 'b']
```

## See also

- [`db.count`](/srvdb/reference/python/db-count/): Return the number of searchable vectors.
- [`db.get`](/srvdb/reference/python/db-get/): Return the metadata stored for an id.
