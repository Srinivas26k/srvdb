---
title: db.neighbors
description: List the outgoing links of a record.
---

List the outgoing links of a record.

```python
db.neighbors(id: str, type: str | None = None) -> list[tuple[str, str]]
```

## Parameters

- **id** : *str*
  An existing record.

- **type** : *str, optional*
  Only links of this type. Default `None` returns every type.

## Returns

`(link_type, target id)` tuples, sorted by type and then target.

## Raises

- **ValueError**
  The id is unknown or `type` is not a valid link type name.

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
vs = [[float(i == j) for j in range(128)] for i in range(3)]
db.add(["a", "b", "c"], vs, ["{}"] * 3)
db.link("a", "cites", "b")
db.link("a", "about", "c")
print(db.neighbors("a"))
# [('about', 'c'), ('cites', 'b')]
print(db.neighbors("a", type="cites"))
# [('cites', 'b')]
```

## See also

- [Links between records](/srvdb/guides/links/): Durability, delete and search expansion.
