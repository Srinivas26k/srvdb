# Library comparison benchmark

Runs SrvDB and other vector-search libraries on the same seeded dataset and
measures the same things as the Rust harness (`crates/srvdb/examples/bench`):
recall@10 against exact brute-force cosine search, QPS, p50/p99 latency,
build time and peak RSS. The result JSON feeds the website page
`/srvdb/benchmarks/` (`website/src/data/benchmarks/comparison.json`).

## Setup (a dedicated venv; competitors are never SrvDB dependencies)

```bash
uv venv -p 3.12 .bench-venv
uv pip install -p .bench-venv/bin/python -r benchmarks/compare/requirements.txt
VIRTUAL_ENV=$PWD/.bench-venv .bench-venv/bin/maturin develop --release
```

## Run

From the repository root, with a clean tree so the commit hash is exact:

```bash
.bench-venv/bin/python benchmarks/compare/run.py --dims 384,768 --ns 5000,50000
```

Options: `--queries` (default 500), `--k` (10), `--seed` (42), `--threads`
(default `1,all`), `--libs` (subset of the registry in `child.py`),
`--work-dir` (dataset cache and temporary indexes), `--out`.

## Method

- Dataset: the Rust harness's SplitMix64 clustered-Gaussian generator,
  exported with `--export-fvecs`, so every library sees identical vectors.
  Vectors are L2-normalised; other libraries use inner product (= cosine).
- HNSW everywhere: M=16, ef_construction=200, ef_search swept over
  10, 16, 32, 64, 128, 256. FAISS IVFFlat: nlist = sqrt(n), nprobe swept.
  SrvDB IVF: nlist = sqrt(n), nprobe = 8 (single point). SQ8 indexes train
  on up to 10 000 vectors. turbovec: TurboQuantIndex at 4 and 2 bits,
  calibrated on a 1024-row random sample.
- Each (library, index, thread count) runs in its own process with
  `OMP_NUM_THREADS`/`RAYON_NUM_THREADS` set. "1 thread" and "all threads"
  runs are labelled in the JSON. `qps` is one query per call on one caller
  thread; `batch_qps` sends all queries in one call.
- Peak RSS is `ru_maxrss` of the child and includes the dataset copy
  (`baseline_rss_mib` is recorded before the build for comparison).
- SrvDB goes through its Python package (list conversion included), the same
  way the other libraries go through theirs. SrvDB build time includes
  `persist()` (durable write to disk); the others build in memory only.

## Licences of compared libraries

| Library | Licence |
|---------|---------|
| faiss-cpu | MIT |
| hnswlib | Apache-2.0 |
| usearch | Apache-2.0 |
| turbovec | MIT |
| numpy | BSD-3-Clause |
| psutil | BSD-3-Clause |
| maturin | MIT or Apache-2.0 |
