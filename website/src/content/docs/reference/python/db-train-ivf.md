---
title: db.train_ivf
description: Train the IVF index on the stored vectors.
---

Train the IVF index on the stored vectors.

```python
db.train_ivf() -> None
```

## Parameters

None.

## Returns

`None`.

## Raises

- **ValueError**
  The database is not in `"ivf"` mode; call `set_mode("ivf")` first.

- **OSError**
  Reading the vectors failed.

- **RuntimeError**
  An internal fault occurred.

## Notes

- Trains on up to the first 10,000 stored vectors, then assigns every vector to a partition. Takes time proportional to the data.
- Thread safety: a database object is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `save`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
db.set_mode("ivf")
db.train_ivf()
```

## See also

- [`db.configure_ivf`](/srvdb/reference/python/db-configure-ivf/): Set the IVF partition parameters.
- [`db.set_mode`](/srvdb/reference/python/db-set-mode/): Switch the index mode of an open database.
