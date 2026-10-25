---
title: Choosing an index mode
description: Flat, HNSW, IVF, SQ8, PQ and auto, and how to choose between them.
---

Every mode ranks by cosine similarity and reports true cosine scores in
[-1, 1]. They differ in how they search and how much they store.

| Mode | Search | Stored per vector | Training |
|------|--------|-------------------|----------|
| `flat` (default) | Exact scan | 4 bytes × dimension | None |
| `hnsw` | Approximate graph | Full vectors + graph | None |
| `ivf` | Partitions, then graph | Full vectors + index | `train_ivf()` |
| `sq8` | Scan of 8-bit codes | 1 byte × dimension | First batch |
| `pq` | Scan of PQ codes | dimension ÷ 8 bytes (÷ 4 if not divisible by 8) | First batch, ≥ 256 vectors |
| `auto` | Flat, then HNSW | Full vectors | None |

## How to choose

1. **Start with `flat`.** Results are exact. It is fast enough for many
   local collections.
2. **Search too slow?** Try `hnsw`, then raise `set_ef_search()` until recall
   is good enough.
3. **Storage too large?** Try `sq8` (4× smaller), then `pq` (up to 32×).
4. **Always measure recall** against `flat` on a sample of your own data.
   Approximate modes vary with the data; PQ in particular can lose accuracy
   on tightly clustered embeddings.

```python
def recall_at_k(exact_db, approx_db, queries, k=10):
    hits = 0
    for q in queries:
        truth = {i for i, _ in exact_db.search(q, k)}
        hits += len(truth & {i for i, _ in approx_db.search(q, k)})
    return hits / (k * len(queries))
```

## Flat

```python
db = srvdb.Database("./db", dimension=768, mode="flat")
```

## HNSW

A layered graph searched by walking from neighbour to neighbour. The graph is
saved by `persist()` and rebuilt from the vectors if missing or damaged.

```python
db = srvdb.Database.new_hnsw("./db", dimension=768, m=16, ef_construction=200, ef_search=50)
db.set_ef_search(120)   # higher: better recall, slower search
```

## IVF

Groups vectors into `nlist` partitions (spherical k-means) and searches only
the `nprobe` closest ones, each with its own HNSW graph.

```python
db = srvdb.Database("./db", dimension=768, mode="ivf")
db.add(ids, embeddings, metadatas)
db.configure_ivf(nlist=100, nprobe=10)
db.train_ivf()
db.persist()            # the trained index is saved and reloaded on open
```

## SQ8

Each value stored as one byte, scaled per dimension from the first batch.

```python
db = srvdb.Database("./db", dimension=768, mode="sq8")
```

## PQ

Each group of 8 (or 4) dimensions stored as one byte: the index of its
nearest learned centroid. Trained on the first batch, which needs at least 256
vectors.

```python
db = srvdb.Database("./db", dimension=768, mode="pq")
```

From Rust, `SrvDB::new_with_hnsw_quantized` builds an HNSW graph over PQ
codes.

## Auto

Starts flat. At each `persist()`, once the database holds at least 50,000
vectors and at least 2 GB of RAM is available, it builds an HNSW graph.
It never compresses: compression is always your choice.

```python
db = srvdb.Database("./db", dimension=768, mode="auto")
```
