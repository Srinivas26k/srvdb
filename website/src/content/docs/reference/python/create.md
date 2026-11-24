---
title: srvdb.create
description: Create a new database with its index and options.
---

Create a new database. Its dimension, index and options are fixed here and recorded on disk, so later you reopen it with only the path.

```python
srvdb.create(path: str, dimension: int, index: str = "hnsw", *, exist_ok: bool = False, **options) -> db
```

## Parameters

- **path** : *str*
  Directory for the database files, used exactly as given. It must not exist, or must be an empty directory. Missing parent directories are created.

- **dimension** : *int*
  Vector dimension, 1 to 4096. Every vector added and every query must have it.

- **index** : *str*, default `"hnsw"`
  One of `"hnsw"`, `"flat"`, `"ivf"`, `"sq8"`, `"pq"`, `"rabitq"` or `"auto"` (case-insensitive). See [Choosing an index mode](/srvdb/guides/index-modes/).

- **exist_ok** : *bool*, default `False`
  If `True` and a database already exists at `path` with the same index, dimension and options, open it instead of raising, like `os.makedirs(exist_ok=True)`. If its settings differ, raise `ValueError`.

- **options** : keyword arguments
  Each index accepts only its own options:

  | Index | Options (default) |
  |-------|-------------------|
  | `hnsw` | `m` (16), `ef_construction` (200), `ef_search` (50) |
  | `ivf` | `nlist` (100), `nprobe` (10), plus `m`, `ef_construction`, `ef_search` for the partition graphs |
  | `sq8`, `pq` | `training`: sample vectors the quantizer is trained on; without it, the first batch added (PQ needs at least 256) |
  | `rabitq` | `bits` (1, 2 or 4; default 4), `rerank` (40, 20 or 10 for 1, 2 or 4 bits; 0 disables), `seed` |
  | `flat`, `auto` | none |

## Returns

The open database object (`db`). Use it in a `with` block to save and close it on exit; see [the database object](/srvdb/reference/python/db/).

## Raises

- **ValueError**
  Unknown `index`; an option that does not apply to the index (the message lists the valid ones); a value out of range (for example `m` outside 2 to 256, `bits` not 1, 2 or 4); `dimension` outside 1 to 4096; training vectors of another dimension; or, with `exist_ok=True`, an existing database with other settings. Nothing is written.

- **FileExistsError**
  `path` is a file, already holds a database, or is a non-empty directory.

- **BlockingIOError**
  The database is open elsewhere.

- **OSError**
  The files cannot be created.

- **RuntimeError**
  An internal fault occurred.

## Notes

- The settings are durable when `create` returns: a crash right after still leaves a database that [`srvdb.open`](/srvdb/reference/python/open/) reopens with the same index.
- Search-time knobs can be changed later: [`db.set_ef_search`](/srvdb/reference/python/db-set-ef-search/), [`db.configure_ivf`](/srvdb/reference/python/db-configure-ivf/), [`db.set_rabitq_rerank`](/srvdb/reference/python/db-set-rabitq-rerank/).
- One database object holds an exclusive lock on the directory until it is closed.

## Examples

```python
import srvdb

with srvdb.create("./vectors", dimension=128) as db:
    db.add(["a"], [[1.0] + [0.0] * 127], ["{}"])
print(srvdb.open("./vectors").count())
# 1
```

```python
import srvdb

db = srvdb.create("./graph", dimension=128, index="hnsw", m=32, ef_search=100)
print(db)
# SrvDB(dimension=128, mode='hnsw', vectors=0)
db.close()
db = srvdb.create("./graph", dimension=128, index="hnsw", exist_ok=True, m=32)
print(db.count())
# 0
```

```python
import srvdb

try:
    srvdb.create("./codes", dimension=128, index="rabitq", m=16)
except ValueError as e:
    print(e)
# invalid input: option 'm' does not apply to index 'rabitq'; valid options: bits, rerank, seed
```

## See also

- [`srvdb.open`](/srvdb/reference/python/open/): Open an existing database by path.
- [`srvdb.delete`](/srvdb/reference/python/delete/): Delete a database from disk.
- [The database object](/srvdb/reference/python/db/): What `create` returns.
