---
title: db.close
description: Release the database and its lock without saving.
---

Release the database and its directory lock without saving.

```python
db.close() -> None
```

## Parameters

None.

## Returns

`None`.

## Raises

Nothing. Calling `close()` on a closed database does nothing.

## Notes

- Writes since the last [`save()`](/srvdb/reference/python/db-save/) are discarded. Call `save()` first to keep them, or use a `with` block, which saves on a clean exit.
- After `close()`, every other method raises `ValueError` ("this Database is closed"), and the directory can be opened again or [deleted](/srvdb/reference/python/delete/).
- Without `close()`, the lock is released when the object is garbage-collected.
- Thread safety: `close()` waits for running calls to finish.

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
db.add(["a"], [[1.0] + [0.0] * 127], ["{}"])
db.save()
db.close()
print(db)
# SrvDB(closed)
print(srvdb.open("./vectors").count())
# 1
```

## See also

- [`db.save`](/srvdb/reference/python/db-save/): Make every earlier write durable (crash-safe commit).
- [The database object](/srvdb/reference/python/db/): The context manager.
