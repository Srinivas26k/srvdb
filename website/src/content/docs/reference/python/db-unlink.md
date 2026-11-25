---
title: db.unlink
description: Remove a link between two records.
---

Remove the link `source -link_type-> target`.

```python
db.unlink(source: str, link_type: str, target: str) -> bool
```

## Returns

`True` if the link existed, otherwise `False` (also for unknown ids).

## Raises

- **ValueError**
  `link_type` is not a valid link type name.

## Notes

Durable after the next `db.save()`.

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
db.add(["a", "b"], [[1.0] + [0.0] * 127, [0.0, 1.0] + [0.0] * 126], ["{}", "{}"])
db.link("a", "cites", "b")
print(db.unlink("a", "cites", "b"))
# True
print(db.neighbors("a"))
# []
```

## See also

- [`db.link`](/srvdb/reference/python/db-link/): Add a link.
