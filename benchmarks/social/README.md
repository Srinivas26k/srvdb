# Memory and recall benchmark (social charts)

Measures, for every SrvDB mode, how many bytes per vector a search scans
and the recall@10 it reaches, on public embedding datasets, then renders
PNGs for X/LinkedIn and the website. Results and the written report are
in [`results.json`](results.json) and [`REPORT.md`](REPORT.md).

Speed (QPS, latency, build time) is deliberately **not** measured here:
those numbers depend on machine load and wait for the idle-machine run
(#54). Everything below is independent of load.

## Setup (a dedicated venv; nothing here is a SrvDB dependency)

```bash
uv venv -p 3.12 .bench-venv
uv pip install -p .bench-venv/bin/python numpy matplotlib h5py pyarrow maturin
VIRTUAL_ENV=$PWD/.bench-venv .bench-venv/bin/maturin develop --release
```

## Run

From the repository root, with a clean tree so the recorded commit is exact:

```bash
.bench-venv/bin/python benchmarks/social/run.py --crash
```

- `--datasets a,b` picks datasets (default: all four, one at a time).
- `--modes` picks modes (default: flat, sq8, pq, rabitq4/2/1 and the three
  RaBitQ modes with rerank off).
- `--crash` also runs the CI kill -9 test locally
  (`SRVDB_CRASH_ITERATIONS=250 cargo test --release -p srvdb --test test_crash`).
- `--no-render` skips the charts; `python benchmarks/social/render.py`
  redraws them from `results.json` alone.

Each mode runs in its own subprocess (so peak RSS is per mode and memory
is freed between runs); `results.json` is rewritten after every mode, so
an interrupted run keeps its partial results.

## Datasets

Downloaded on first use into `benchmarks/social/.data/` (gitignored),
checked against the size the server reports (and the Hugging Face
SHA256), and recorded with their SHA256 in `results.json`. Downloads live
only in `benchmarks/`; the SrvDB library never touches the network.

| Name | Source | Size used | Ground truth | Licence |
|---|---|---|---|---|
| `dbpedia-openai-100k` | [KShivendu/dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M), OpenAI ada-002, 1536-d | first 100,000 rows as base, next 1,000 as queries | exact cosine, numpy brute force | MIT (dataset card) |
| `nytimes-256-angular` | [ann-benchmarks](http://ann-benchmarks.com) HDF5, 256-d | all 290,000 base, first 1,000 test queries | ann-benchmarks `neighbors` | derived from UCI Bag of Words (CC BY 4.0) |
| `glove-100-angular` | [ann-benchmarks](http://ann-benchmarks.com) HDF5, 100-d | all 1,183,514 base, first 1,000 test queries | ann-benchmarks `neighbors` | GloVe, PDDL 1.0 |
| `synthetic-768` | seeded clustered Gaussian (numpy, seed 42), offline | 50,000 base, 1,000 queries | exact cosine, numpy brute force | generated |

`sift-128-euclidean` is not run: SrvDB searches by cosine similarity
only, and SIFT's ground truth is Euclidean. Vectors are L2-normalised
before insertion; on normalised vectors cosine and angular rankings agree.

## What is measured

- **Search code bytes per vector**: `db.memory_report()["vector_bytes"] / n`,
  the bytes per vector the scan reads (float32 for flat, codes otherwise).
  This is what the charts show.
- **On-disk bytes by file role** (`full_vectors`, `codes`, `quantizer`,
  `ids_metadata`, `other`). Vector and code files are preallocated to a
  capacity larger than `n`, so on-disk bytes per vector include growth
  slack. RaBitQ also keeps the float32 vectors on disk for exact rerank
  (indexes are derived data), so its total on disk is larger than flat's.
- **Recall@10** against the ground truth above, default settings, plus
  RaBitQ with `set_rabitq_rerank(0)`.
- **Peak RSS** of the measuring process (`ru_maxrss`), which includes the
  benchmark's own copies of the base and query arrays (`input_arrays_mib`).

## Competitors (separate venv, never SrvDB dependencies)

```bash
uv venv -p 3.12 .competitor-venv
uv pip install -p .competitor-venv/bin/python numpy==2.5.5 faiss-cpu==1.16.0 turbovec==1.1.2
for d in nytimes-256-angular glove-100-angular dbpedia-openai-100k synthetic-768; do
  .competitor-venv/bin/python benchmarks/social/competitors.py $d
done
.bench-venv/bin/python benchmarks/social/render.py
.bench-venv/bin/python benchmarks/social/report.py
```

Runs on the arrays `run.py` cached, so it must come after `run.py`.
FAISS `IndexRaBitQ` 4/2/1-bit with and without float32 refine (k_factor
10/20/40, SrvDB's defaults), `IndexScalarQuantizer` QT_8bit, `IndexPQ`
(M = largest divisor of dim <= dim/8), OPQ+PQ for dim <= 256, sign bits in
`IndexBinaryFlat` with and without rerank (pgvector-style), and turbovec
`TurboQuantIndex` 4/2-bit (needs dim divisible by 8, so not GloVe-100).
Code bytes are FAISS `sa_code_size()` and turbovec `len(to_bytes()) / n`.
faiss-cpu and turbovec are MIT.

## Files

- `run.py` orchestrates, records the environment, writes `results.json`.
- `measure.py` is the per-dataset / per-mode child process.
- `datasets.py` downloads, verifies and loads datasets.
- `competitors.py` runs FAISS and turbovec on the same cached data.
- `render.py`, `charts.py`, `compare_chart.py` draw `img/*.png` from
  `results.json` only.
- `report.py` + `findings.py` write `REPORT.md` from `results.json` only.
