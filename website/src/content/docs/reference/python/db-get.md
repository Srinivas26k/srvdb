---
title: srvdb.Database.get
description: Return the metadata stored for an id.
---

Return the metadata stored for an id.

```python
Database.get(id: str) -> str | None
```

## Parameters

- **id** : *str*
  The vector's id.

## Returns

`str`: the metadata as a JSON object string, including the id under the key `"__id__"`. Parse it with `json.loads`. The stub type allows `None`, which the core returns only if the metadata entry itself is missing.

## Raises

- **ValueError**
  No live vector has this id (it never existed or was deleted).

- **OSError**
  Reading the metadata failed.

- **RuntimeError**
  An internal fault occurred.

## Notes

- Thread safety: releases the GIL and takes a shared lock, so calls from several threads run in parallel; they wait while a write is running. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import json
import srvdb

db = srvdb.open("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a"], [v], [json.dumps({"title": "A"})])
print(json.loads(db.get("a")))
# {'__id__': 'a', 'title': 'A'}
```

## See also

- [`srvdb.Database.get_all_ids`](/srvdb/reference/python/database-get-all-ids/): Return every live id.
- [`srvdb.Database.search`](/srvdb/reference/python/database-search/): Find the k nearest vectors to a query.
