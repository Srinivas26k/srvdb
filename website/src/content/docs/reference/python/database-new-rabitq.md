---
title: srvdb.Database.new_rabitq
description: Open or create a database searched through 1, 2 or 4-bit RaBitQ codes with exact rerank.
---

Open or create a database searched through 1, 2 or 4-bit RaBitQ codes with exact rerank.

```python
srvdb.Database.new_rabitq(path: str, dimension: int = 1536, bits: int = 4, rerank: int | None = None, seed: int | None = None) -> srvdb.Database
```

## Parameters

- **path** : *str*
  Directory holding the database files.

- **dimension** : *int*
  Vector dimension, 1 to 4096. Must match the stored dimension when the database exists.

- **bits** : *int*
  Bits per dimension in the codes: 1, 2 or 4.

- **rerank** : *int or None*
  Candidates per result rescored against the full vectors; `0` returns estimated scores without rerank. `None` uses 40, 20 or 10 for 1, 2 or 4 bits.

- **seed** : *int or None*
  Seed of the random rotation. `None` uses a fixed default. Opening an existing database with another seed or bit width rebuilds the codes.

## Returns

`Database` in `"rabitq1"`, `"rabitq2"` or `"rabitq4"` mode.

## Raises

- **ValueError**
  `bits` is not 1, 2 or 4, `rerank` is above 1 000 000, `dimension` is 0 or above 4096, or it does not match the stored database.

- **BlockingIOError**
  The database is already open elsewhere.

- **OSError**
  The files cannot be created or read.

- **RuntimeError**
  Stored vectors or metadata are damaged, or an internal fault occurred. Damaged code files are rebuilt instead.

## Notes

- No training: vectors can be added one at a time from the start.
- Full vectors stay on disk; the codes are derived data, rebuilt on open when missing, damaged or made with other settings. See [RaBitQ low-bit search](/srvdb/guides/rabitq/).
- `rerank` is recorded by `persist()`; a value passed on open replaces it, or call [`set_rabitq_rerank`](/srvdb/reference/python/database-set-rabitq-rerank/).
- Thread safety: a `Database` is safe to share between threads. This method releases the GIL; writes take an exclusive lock and wait for running searches instead of failing. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import random
import srvdb

rng = random.Random(0)
vectors = [[rng.random() - 0.5 for _ in range(128)] for _ in range(50)]
db = srvdb.Database.new_rabitq("./vectors", 128, bits=2)
db.add([f"v{i}" for i in range(50)], vectors, ["{}"] * 50)
print(db.search(vectors[7], k=1)[0][0])
# v7
```

## See also

- [`srvdb.Database.set_rabitq_rerank`](/srvdb/reference/python/database-set-rabitq-rerank/): Change the rerank factor of a RaBitQ database.
- [`srvdb.open`](/srvdb/reference/python/open/): Open a database, creating it if it does not exist.
