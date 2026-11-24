# 02 Index modes

Builds the same dataset (`make_blobs`, 128 dimensions, 50 clusters;
30,000 vectors, or 3,000 with `--fast`) in six databases:
`index="flat"`, `"hnsw"` (`ef_search=64`), `"ivf"` (`nlist=64`,
`nprobe=8`, then `train_ivf()`), `"sq8"`, `"pq"` (both trained on the
first 10,000 vectors, or all 3,000 with `--fast`) and `"rabitq"` (4 bits).
For each it measures:

- recall@10 against exact (brute-force NumPy) search,
- single-query throughput (queries per second, one `db.search` at a time),
- estimated resident memory from `db.memory_report()["total_bytes"]`,
- build time,
- the share of the true top 10 found anywhere in the top 100 (table only),
  which tells a coarse ranking apart from neighbours that are lost.

```bash
python examples/gallery/02_index_modes/index_modes.py [--fast]
```

Output: a table (also written as `output/02_index_modes.csv`) and
`output/02_index_modes.png`, three bar panels (recall, QPS, memory) with
one bar per mode, each labelled with its value; every axis starts at zero.

How to read it: flat is exact and the reference. Graph and partition
indexes (HNSW, IVF) buy speed with memory; quantizers (SQ8, PQ, RaBitQ)
buy memory with some recall, which RaBitQ recovers by reranking against
the full vectors. Measure on your own embeddings before choosing.

**Why PQ recall is low here.** SrvDB's PQ has no sub-space setting: it
always uses `dimension / 8` sub-spaces (16 at 128 dimensions, so 16 bytes
per vector, 32 times less than flat) with 256 centroids each, and it
keeps no full vectors to rerank with. In `make_blobs` data the true
neighbours of a query sit in one cluster and differ from the other
members of that cluster only by small random noise, which 16 one-byte
codes cannot resolve, so PQ ranks the whole cluster almost equally and
recall@10 is low. It is not broken: the top-100 column shows how many
true neighbours it still places near the top. The limit is the code
size, not the training sample. Real embeddings
are less uniform inside a cluster; if you need PQ-level memory with high
recall, RaBitQ (which reranks) is usually the better choice. See the
[index modes guide](https://srinivas26k.github.io/srvdb/guides/index-modes/).
