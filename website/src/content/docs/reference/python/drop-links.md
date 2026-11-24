---
title: srvdb.drop_links
description: Delete every link of a closed database after an older SrvDB changed it.
---

Delete every link between records of a closed database.

```python
srvdb.drop_links(path: str) -> int
```

## Parameters

- **path** : *str*
  Directory of a closed database.

## Returns

The number of links removed.

## Raises

- **FileNotFoundError**
  No database at `path`.

- **BlockingIOError**
  The database is open, in this process or another. Close it first.

- **OSError**
  A file could not be read or written.

## Notes

- Use it when `srvdb.open` raises `RuntimeError` saying the links were written before an older SrvDB (0.3 or earlier, which ignores links) added, deleted or compacted records. Those links may point at the wrong records, so SrvDB refuses to open the database rather than use them or drop them silently.
- Vectors and metadata are not changed. Link the records again afterwards.
- It never runs on its own.

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
db.add(["a", "b"], [[1.0] + [0.0] * 127, [0.0, 1.0] + [0.0] * 126], ["{}", "{}"])
db.link("a", "cites", "b")
db.save()
db.close()

print(srvdb.drop_links("./vectors"))
# 1
db = srvdb.open("./vectors")
print(db.neighbors("a"), db.count())
# [] 2
```

## See also

- [Links between records](/srvdb/guides/links/): Durability and compatibility.
