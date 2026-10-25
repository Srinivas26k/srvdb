# Indexing Modes

SrvDB lets you choose how vectors are stored and searched. Every mode
ranks by cosine similarity and reports true cosine scores in [-1, 1].
Approximate and compressed modes trade **recall** (accuracy) for speed or
size; how much depends on your data, so measure recall on a sample of your
own embeddings against `flat` before choosing one.

## Summary

| Mode | Search | Stored per vector | Training |
|------|--------|-------------------|----------|
| **flat** (default) | Exact scan | 4 bytes × dimension | None |
| **hnsw** | Approximate graph | Full vectors + graph | None |
| **ivf** | Partitions, then graph | Full vectors + index | `train_ivf()` |
| **sq8** | Scan of 8-bit codes | 1 byte × dimension | First batch |
| **pq** | Scan of PQ codes | dimension ÷ 8 bytes (÷ 4 if not divisible by 8) | First batch, at least 256 vectors |
| **auto** | Flat, then HNSW | Full vectors | None |

No performance figures are listed here: SrvDB publishes numbers only from
a committed, reproducible benchmark. See [Benchmarking](../guides/benchmarking.md).

---

## 1. Flat (default)

Stores full-precision (`f32`) vectors and scans all of them with SIMD
cosine similarity. Results are exact.

- **Pros**: Exact results. No training. Simple.
- **Cons**: Search time grows linearly with the number of vectors.

```python
db = srvdb.Database("path/to/db", dimension=1536, mode="flat")
```

## 2. HNSW graph

[Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320)
links vectors into a layered graph and searches by walking it, visiting a
small part of the data. The graph is saved by `persist()` and rebuilt from
the vectors if it is missing or damaged.

- **Pros**: Search visits far fewer vectors than a scan on large data.
- **Cons**: Approximate. Extra memory for the graph; inserts cost more.

```python
db = srvdb.Database.new_hnsw(
    "path/to/hnsw_db",
    dimension=1536,
    m=16,                # links per node
    ef_construction=200, # build-time candidate list
    ef_search=50,        # search-time candidate list
)
db.set_ef_search(100)    # higher: better recall, slower search
```

## 3. Product quantization (PQ)

Splits each vector into sub-vectors (8 dimensions each, or 4 when the
dimension is not divisible by 8) and stores one byte per sub-vector: the
index of its nearest learned centroid. Codebooks are trained on the first
batch added, which must hold at least 256 vectors. In Python, `pq` mode
scans the codes; from Rust, `SrvDB::new_with_hnsw_quantized` builds an HNSW
graph over them.

- **Pros**: Up to 32× smaller than `f32` storage.
- **Cons**: Approximate scores. Accuracy can drop sharply on tightly
  clustered embeddings (many documents on the same topic).

```python
db = srvdb.Database("path/to/pq_db", dimension=1536, mode="pq")
# or, training on a sample you choose:
db = srvdb.Database.new_product_quantized(
    "path/to/pq_db", dimension=1536, training_vectors=sample_vectors
)
```

## 4. Scalar quantization (SQ8)

Stores each value as an 8-bit integer, scaled per dimension using the first
batch added.

- **Pros**: 4× smaller than `f32` storage.
- **Cons**: Approximate scores; still a linear scan.

```python
db = srvdb.Database("path/to/sq8_db", dimension=1536, mode="sq8")
```

## 5. IVF (inverted file)

Groups vectors into `nlist` partitions around centroids (spherical k-means,
matching cosine search). A query searches only the `nprobe` closest
partitions, each with its own HNSW graph. The trained index is saved by
`persist()` and reloaded on open.

- **Pros**: Searches a fraction of the data on large collections.
- **Cons**: Needs training after data is loaded; approximate.

```python
db = srvdb.Database("path/to/ivf_db", dimension=1536, mode="ivf")
db.add(ids, embeddings, metadatas)
db.configure_ivf(nlist=100, nprobe=10)
db.train_ivf()
db.persist()
```

## 6. Auto

Starts as flat. At each `persist()` it checks the database size and
available memory, and builds an HNSW graph once the database is large. It
never compresses vectors: compression is always your explicit choice.

```python
db = srvdb.Database("path/to/db", dimension=1536, mode="auto")
```
