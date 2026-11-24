---
title: db.link
description: Add a typed, directed link between two records.
---

Add the directed link `source -link_type-> target`.

```python
db.link(source: str, link_type: str, target: str) -> bool
```

## Parameters

- **source**, **target** : *str*
  Ids of existing records; they must differ.

- **link_type** : *str*
  A lowercase letter, then up to 63 lowercase letters, digits or `_`.

## Returns

`True` if the link is new, `False` if it already existed.

## Raises

- **ValueError**
  An id is unknown, `source` equals `target`, `link_type` is not a valid name, or a limit would be exceeded: 1,024 outgoing links for `source`, 100,000 incoming links for `target`, 1,000,000 links or 1,024 link type names per database.

## Notes

- Visible at once; durable after `db.save()`, together with the records saved with it.
- Deleting either record removes the link; `upsert` keeps it on the new version.

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
db.add(["a", "b"], [[1.0] + [0.0] * 127, [0.0, 1.0] + [0.0] * 126], ["{}", "{}"])
print(db.link("a", "cites", "b"))
# True
print(db.link("a", "cites", "b"))
# False
```

## See also

- [Links between records](/srvdb/guides/links/): Durability, delete and search expansion.
- [`db.unlink`](/srvdb/reference/python/db-unlink/): Remove a link.
- [`db.neighbors`](/srvdb/reference/python/db-neighbors/): List a record's links.
