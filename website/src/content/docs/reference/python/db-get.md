---
title: db.get
description: Return the metadata stored for an id.
---

Return the metadata stored for an id.

```python
db.get(id: str) -> str | None
```

## Parameters

- **id** : *str*
  The vector's id.

## Returns

`str | None`: the metadata as a JSON object string, including the id under the key `"__id__"` (parse it with `json.loads`), or `None` if no live vector has this id: it never existed or was deleted. Like `dict.get`, an unknown id is not an error.

## Raises

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

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a"], [v], [json.dumps({"title": "A"})])
print(json.loads(db.get("a")))
# {'__id__': 'a', 'title': 'A'}
print(db.get("missing"))
# None
```

## See also

- [`db.get_all_ids`](/srvdb/reference/python/db-get-all-ids/): Return every live id.
- [`db.search`](/srvdb/reference/python/db-search/): Find the k nearest vectors to a query.
