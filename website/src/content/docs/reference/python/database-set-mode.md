---
title: srvdb.Database.set_mode
description: Switch the index mode of an open database.
---

Switch the index mode of an open database.

```python
Database.set_mode(mode: str) -> None
```

## Parameters

- **mode** : *str*
  Case-insensitive: `"flat"`, `"hnsw"`, `"sq8"` (alias `"scalar"`), `"ivf"` or `"auto"`.

## Returns

`None`.

## Raises

- **ValueError**
  Unknown mode, including `"pq"`, which needs [`new_product_quantized`](/srvdb/reference/python/database-new-product-quantized/).

## Notes

- The mode is recorded by the next `persist()`; later opens use it.
- After `set_mode("ivf")`, call [`train_ivf`](/srvdb/reference/python/database-train-ivf/).
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.open("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a"], [v], ["{}"])
db.set_mode("hnsw")
print(db.info())
# SrvDB(dimension=128, mode='hnsw', vectors=1)
```

## See also

- [`srvdb.open`](/srvdb/reference/python/open/): Open a database, creating it if it does not exist.
- [`srvdb.Database.train_ivf`](/srvdb/reference/python/database-train-ivf/): Train the IVF index on the stored vectors.
- [`srvdb.Database.set_ef_search`](/srvdb/reference/python/database-set-ef-search/): Set HNSW's search candidate list size.
