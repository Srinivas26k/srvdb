# Benchmarks

SrvDB's benchmark harness is the baseline for all performance work from 0.4
on. Any published number must come from this harness (or another committed,
reproducible benchmark) and list the hardware, OS, dataset, commit hash and
command, which the harness records for you.

## What it measures

For each mode (`flat`, `sq8`, `pq`, `hnsw`, `ivf`):

| Metric | How |
|--------|-----|
| recall@k | Mean overlap between the mode's top-k and the exact top-k from brute-force cosine search (SrvDB's metric) over the same base vectors. |
| QPS | Queries divided by wall time, one query at a time on one caller thread (after 10 warm-up queries). The engine may use internal parallelism. |
| p50 / p99 latency | Nearest-rank percentiles of per-query wall time. |
| build time | Opening the database, inserting all vectors in batches of 1000, training (SQ8/PQ quantizers on up to 10 000 vectors; IVF k-means), and `persist()`. |
| peak RSS | `VmHWM` from `/proc/self/status` (Linux only, `n/a` elsewhere). |

Each mode runs in a separate child process, so peak RSS is per mode. It
includes the in-memory copy of the dataset (`n * dim * 4` bytes) that the
harness holds, which is the same for every mode; compare modes against each
other rather than reading it as the database's footprint alone.

Databases are created in a temporary directory and deleted afterwards.

## Running it

Always build in release mode; debug numbers are meaningless and are marked
`debug` in the output.

```bash
# Default: synthetic 384-dim clustered Gaussians, 10 000 vectors, 200 queries
cargo run --release -p srvdb --example bench

# 768 dims, larger, selected modes only
cargo run --release -p srvdb --example bench -- --dim 768 --n 50000 --modes flat,hnsw,ivf

# Show all options
cargo run --release -p srvdb --example bench -- --help
```

The synthetic dataset is deterministic: the same `--dim`, `--n`, `--queries`
and `--seed` produce the same vectors on every machine. It is generated in
memory, so the harness works offline and in CI.

Useful options: `--k` (default 10), `--ef-search` (HNSW and IVF partition
search list, default 100), `--nlist` (IVF partitions, default `sqrt(n)`),
`--nprobe` (default 8), `--work-dir` (where temporary databases go),
`--out` (JSON path, default `bench-results.json`, git-ignored).

## Real datasets (SIFT1M, GloVe)

The library never downloads anything and datasets are never committed. To
use a public dataset, download it yourself and pass the `.fvecs` files:

```bash
cargo run --release -p srvdb --example bench -- \
  --base-fvecs /data/sift/sift_base.fvecs \
  --query-fvecs /data/sift/sift_query.fvecs \
  --n 1000000 --queries 1000
```

`--n` and `--queries` cap how many vectors are read. Ground truth is always
recomputed by brute force with cosine similarity (the dataset's own L2
ground-truth files are not used), so it matches SrvDB's metric. GloVe is
distributed as text; convert it to `.fvecs` first. Brute-force ground truth
on 1M vectors takes a while and is repeated per mode.

## Output

The harness prints a Markdown table to stdout, preceded by the CPU model,
logical core count, OS, commit hash (with `-dirty` if the tree has
uncommitted changes), dataset description and the exact command. The same
data is written as JSON:

```json
{
  "environment": { "cpu_model": "...", "logical_cores": 8, "os": "...",
                   "commit": "...", "srvdb_version": "0.3.0",
                   "profile": "release", "command": "cargo run --release ..." },
  "dataset": { "name": "synthetic-gauss(...)", "dim": 384, "base": 10000,
               "queries": 200, "k": 10, "seed": 42, ... },
  "results": [ { "mode": "flat", "recall_at_k": 1.0, "qps": 0.0,
                 "p50_ms": 0.0, "p99_ms": 0.0, "build_secs": 0.0,
                 "peak_rss_mib": 0.0 } ]
}
```

A mode that fails is reported with an `error` field and the harness exits
non-zero.

## Publishing numbers

Paste the full table including the header lines into the release notes or
report, and commit the JSON alongside it if it backs a claim. Do not quote a
single number without its environment block.

The micro-benchmark in `crates/srvdb/benches/search_benchmark.rs`
(`cargo bench -p srvdb`) remains for quick flat-search regression checks.
