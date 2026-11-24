---
title: srvdb.Database.new_product_quantized
description: Open or create a PQ database trained on a sample you choose.
---

Open or create a PQ database trained on a sample you choose.

```python
srvdb.Database.new_product_quantized(path: str, dimension: int, training_vectors: Sequence[Sequence[float]]) -> srvdb.Database
```

## Parameters

- **path** : *str*
  Directory holding the database files.

- **dimension** : *int*
  Vector dimension; every training vector must have this many values.

- **training_vectors** : *sequence of sequences of float*
  Sample the codebooks are trained on; 1000 or more is recommended. The sample is not added to the database.

## Returns

`Database` in `"pq"` mode.

## Raises

- **ValueError**
  `training_vectors` is empty, or a training vector does not have `dimension` values.

- **BlockingIOError**
  The database is already open elsewhere.

- **OSError**
  The files cannot be created or read.

- **RuntimeError**
  Stored data is damaged or an internal fault occurred.

## Notes

- PQ stores each vector as a short code. Scores are approximations of cosine similarity and can fall slightly outside [-1, 1].
- `set_mode` cannot switch to or from PQ; choose it when creating the database.
- Unlike the main constructor, `dimension` is not checked against the 128 to 4096 range here.
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes (`add`, `upsert`, `delete`, `persist`, `compact`, mode and IVF changes) take an exclusive lock and wait for each other and for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import random
import srvdb

rng = random.Random(0)
sample = [[rng.random() for _ in range(128)] for _ in range(20)]
db = srvdb.Database.new_product_quantized("./vectors", 128, sample)
db.add(["a"], [sample[0]], ["{}"])
print(db.search(sample[0], k=1)[0][0])
# a
```

## See also

- [`srvdb.Database.new_scalar_quantized`](/srvdb/reference/python/database-new-scalar-quantized/): Open or create an SQ8 database trained on a sample you choose.
- [`srvdb.open`](/srvdb/reference/python/open/): Open a database, creating it if it does not exist.
