# Benchmarks

SrvDB's benchmark harness is the baseline for all performance work from 0.4
on. Any published number must come from this harness (or another committed,
reproducible benchmark) and list the hardware, OS, dataset, commit hash and
command, which the harness records for you.

## What it measures

For each mode (`flat`, `sq8`, `pq`, `hnsw`, `ivf`, `rabitq1`, `rabitq2`,
`rabitq4`):

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
`--nprobe` (default 8), `--rerank` (RaBitQ rerank factor, default 40/20/10
for 1/2/4 bits), `--work-dir` (where temporary databases go),
`--out` (JSON path, default `bench-results.json`, git-ignored).

## Filtered search

`--filter-pct P` restricts every query to a seeded random P% of the base
vectors (independent of the clusters) and reports, per mode, recall@k and
QPS of `search_filtered` against exact filtered search, next to
post-filtering (`search` k, then drop non-matching results). The `short`
columns count queries that returned fewer than k results.

```bash
for p in 1 10 50; do
  cargo run --release -p srvdb --example bench -- \
    --dim 128 --n 50000 --modes flat,sq8,hnsw --filter-pct $p
done
```

Example run on an AMD Ryzen 5 3550H (8 logical cores), Ubuntu 26.04.1,
kernel 7.0.0-43, release build of the code committed as 3a54d1d (before
`EXACT_COST_RATIO` changed from 15 to 4, so HNSW took the exact path at
all three selectivities), synthetic dataset
`synthetic-gauss(dim=128,n=50000,clusters=500,seed=42)`, 200 queries,
k = 10, ef 100. The machine was shared with other builds (load average
above 8), so QPS is indicative only; recall and short counts are not
affected by load. Command, once per P in 1, 10, 50:
`cargo run --release -p srvdb --example bench -- --dim 128 --n 50000 --modes flat,sq8,hnsw --filter-pct P`.

| mode | filter | recall@10 | short | post-filter recall@10 | post-filter short |
|---|--:|--:|--:|--:|--:|
| flat | 1% | 1.000 | 0 | 0.008 | 200 |
| sq8 | 1% | 0.988 | 0 | 0.0075 | 200 |
| hnsw | 1% | 1.000 | 0 | 0.008 | 200 |
| flat | 10% | 1.000 | 0 | 0.0945 | 200 |
| sq8 | 10% | 0.992 | 0 | 0.095 | 200 |
| hnsw | 10% | 1.000 | 0 | 0.0945 | 200 |
| flat | 50% | 1.000 | 0 | 0.504 | 199 |
| sq8 | 50% | 0.985 | 0 | 0.502 | 200 |
| hnsw | 50% | 1.000 | 0 | 0.504 | 199 |

### Exact-scan threshold

With HNSW, `search_filtered` scores the matching ids exactly instead of
walking the graph when `matches^2 < EXACT_COST_RATIO * ef * M * live`
(`crates/srvdb/src/engine/filter.rs`). Equal cost at selectivity `s` is
`s * n * c_exact = c_walk(s)`, so `ratio = s * c_walk / (c_exact * ef * M)`.
An ignored test measures it with the walk forced:

```bash
cargo test --release -p srvdb filtered_walk_cost -- --ignored --nocapture
```

Two consecutive runs on the machine above (Ubuntu 26.04.1, kernel
7.0.0-43, commit 08801ab, load average about 8), with 20 000 seeded
clustered 64-d vectors (200 clusters), 50 queries, M 16, ef 100:

| matching | visits/query | walk us/query | us/visit | exact us/match | ratio |
|--:|--:|--:|--:|--:|--:|
| 100% | 760 / 759 | 331 / 294 | 0.44 / 0.39 | 0.15 / 0.17 | 1.4 / 1.1 |
| 50% | 2384 / 2382 | 1220 / 1205 | 0.51 / 0.51 | 0.17 / 0.20 | 2.2 / 1.9 |
| 10% | 7804 / 7801 | 9415 / 9587 | 1.21 / 1.23 | 0.21 / 0.15 | 2.8 / 4.1 |
| 1% | 19143 / 19141 | 76542 / 74800 | 4.00 / 3.91 | 0.14 / 0.14 | 3.5 / 3.4 |

Each visit gets more expensive as the filter gets more selective, because
of the two-hop expansion. `EXACT_COST_RATIO` is 4, the top of the
selective range where the choice matters. With the defaults (ef 50, M 16)
that means an exact scan below about 18% matching at 100k vectors, 5.7%
at 1M and 1.8% at 10M. A fixed percentage would be too low for small
databases and too high for large ones.

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

## Comparison with other libraries

`benchmarks/compare/` runs SrvDB (through its Python package), FAISS,
hnswlib, usearch and turbovec on the dataset this harness generates
(exported with `--export-fvecs DIR`) and measures the same metrics. The
competitors are installed only in a dedicated venv from
`benchmarks/compare/requirements.txt`; see `benchmarks/compare/README.md`
for setup, method and licences. Regenerate the website data with a clean
tree:

```bash
.bench-venv/bin/python benchmarks/compare/run.py --dims 384,768 --ns 5000,50000
```

This rewrites `website/src/data/benchmarks/comparison.json`, which the
page `/srvdb/benchmarks/` renders at build time.

The runner records `/proc/loadavg` before and after every child run
(`loadavg_before`, `loadavg_after`) and warns when the 1-minute load is
above 1. Pass `--preliminary` for any run on a shared or busy machine; the
JSON then carries `"preliminary": true` and the page shows a notice.

Final run for publication (idle machine: close other programs, no other
builds, laptop on mains power; check `cat /proc/loadavg` reads below 1
first, and check no load warnings appear and every `loadavg_before[0]`
in the JSON is below 1 afterwards):

```bash
git status --short        # must be empty so the commit hash is exact
CARGO_TARGET_DIR=$PWD/target VIRTUAL_ENV=$PWD/.bench-venv .bench-venv/bin/maturin develop --release
.bench-venv/bin/python benchmarks/compare/run.py --dims 384,768 --ns 5000,50000 --queries 500 --k 10 --seed 42 --threads 1,all
```

The micro-benchmark in `crates/srvdb/benches/search_benchmark.rs`
(`cargo bench -p srvdb`) remains for quick flat-search regression checks.
