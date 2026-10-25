---
title: Python API
description: Every class and method of the srvdb Python package.
---

```python
import srvdb
```

Type stubs ship with the package, so editors show signatures and types.

## `srvdb.Database`

### Constructors

```python
srvdb.Database(path: str, dimension: int = 1536, mode: str = "flat")
```

Create or open the database at `path`. `mode` is one of `"flat"`, `"hnsw"`,
`"ivf"`, `"sq8"` (alias `"scalar"`), `"pq"` (alias `"product"`), `"auto"`.
`dimension` must be 128–4096 and match the stored data.

```python
srvdb.Database.new_hnsw(path, dimension=1536, m=16, ef_construction=200, ef_search=50)
srvdb.Database.new_scalar_quantized(path, dimension, training_vectors)
srvdb.Database.new_product_quantized(path, dimension, training_vectors)
```

HNSW with explicit parameters; SQ8 or PQ trained on a sample you choose.

### Writing

| Method | Returns | Description |
|--------|---------|-------------|
| `add(ids, embeddings, metadatas)` | `int` | Add new vectors. Ids must be new and unique; metadata a JSON object string. |
| `upsert(ids, embeddings, metadatas)` | `int` | Insert or replace. |
| `delete(ids)` | `int` | Delete; returns how many existed. Unknown ids are ignored. |
| `persist()` | `None` | Crash-safe commit of everything written so far. |
| `compact()` | `int` | Remove deleted vectors from disk; returns how many. |

Writes are visible to search immediately and durable after `persist()`.

### Reading

| Method | Returns | Description |
|--------|---------|-------------|
| `search(query, k)` | `list[tuple[str, float]]` | `k` nearest ids with cosine scores, best first. |
| `search_batch(queries, k)` | `list[list[tuple[str, float]]]` | Several searches, run in parallel. |
| `get(id)` | `str \| None` | Stored metadata (JSON) for `id`. |
| `count()` | `int` | Number of searchable (not deleted) vectors. |
| `get_all_ids()` | `list[str]` | Every id. |
| `info()` | `str` | Short summary: dimension, mode, size. |

### Tuning

| Method | Description |
|--------|-------------|
| `set_mode(mode)` | Switch to `"flat"`, `"hnsw"`, `"sq8"`, `"ivf"` or `"auto"`. |
| `set_ef_search(n)` | HNSW candidate list size (≥ 1): higher is more accurate, slower. Saved by `persist()`. |
| `configure_ivf(nlist=100, nprobe=10)` | IVF partitions and how many to search. |
| `train_ivf()` | Train IVF on the stored vectors. |

### Threads and errors

Every method releases the GIL; a `Database` is safe to share between threads.
See [Threads and processes](/srvdb/guides/concurrency/) and
[Handling errors](/srvdb/guides/errors/).

## `srvdb.__version__`

The installed version string.
