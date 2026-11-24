# 02 Index modes

Builds the same dataset (`make_blobs`, 128 dimensions, 50 clusters;
30,000 vectors, or 3,000 with `--fast`) in six databases:
`index="flat"`, `"hnsw"` (`ef_search=64`), `"ivf"` (`nlist=64`,
`nprobe=8`, then `train_ivf()`), `"sq8"`, `"pq"` (both trained on the
first 2,000 vectors) and `"rabitq"` (4 bits). For each it measures:

- recall@10 against exact (brute-force NumPy) search,
- single-query throughput (queries per second, one `db.search` at a time),
- estimated resident memory from `db.memory_report()["total_bytes"]`,
- build time.

```bash
python examples/gallery/02_index_modes/index_modes.py [--fast]
```

Output: a table (also written as `output/02_index_modes.csv`) and
`output/02_index_modes.png`, three bar panels (recall, QPS, memory) with
one bar per mode.

How to read it: flat is exact and the reference. Graph and partition
indexes (HNSW, IVF) buy speed with memory; quantizers (SQ8, PQ, RaBitQ)
buy memory with some recall, which RaBitQ recovers by reranking against
the full vectors. On clustered synthetic data most modes reach high
recall; PQ with default settings shows the cost of coarse codes. Measure
on your own embeddings before choosing. See the
[index modes guide](https://srinivas26k.github.io/srvdb/guides/index-modes/).
