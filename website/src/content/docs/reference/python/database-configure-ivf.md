---
title: srvdb.Database.configure_ivf
description: Set the IVF partition parameters.
---

Set the IVF partition parameters.

```python
Database.configure_ivf(nlist: int = 100, nprobe: int = 10) -> None
```

## Parameters

- **nlist** : *int*, default `100`
  Number of partitions (centroids).

- **nprobe** : *int*, default `10`
  Partitions searched per query: higher is more accurate, slower.

## Returns

`None`.

## Raises

- **OverflowError**
  A value is negative.

## Notes

- Call before `set_mode("ivf")` so the index is created with these values. In `"ivf"` mode it replaces the index with an untrained one. Either way, call [`train_ivf`](/srvdb/reference/python/database-train-ivf/) afterwards.
- Values are not range-checked; use positive values with `nprobe <= nlist`.
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.open("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
db.configure_ivf(nlist=2, nprobe=1)
db.set_mode("ivf")
db.train_ivf()
print(db.search(v, k=1))
# [('a', 0.9999999403953552)]
```

## See also

- [`srvdb.Database.train_ivf`](/srvdb/reference/python/database-train-ivf/): Train the IVF index on the stored vectors.
- [`srvdb.Database.set_mode`](/srvdb/reference/python/database-set-mode/): Switch the index mode of an open database.
