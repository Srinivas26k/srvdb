# Benchmarking

Performance varies significantly depending on CPU (Intel vs ARM, AVX-512 support), RAM speed, and disk I/O. We include a universal benchmark script to help you validate SrvDB's performance on your specific hardware.

## Running the Benchmark

SrvDB comes with a benchmark script that:
1. Generates synthetic data (Random + Clustered mix).
2. Indexes it using different modes (Flat, HNSW).
3. Measures Ingestion Speed, Search Latency (P50, P99), and Recall.

### Step 1: Install Dependencies
```bash
pip install "srvdb[bench]"
```

### Step 2: Run the Script
From a clone of the repository: `python benchmarks/universal_benchmark.py`.
Without a clone:
```bash
curl -O https://raw.githubusercontent.com/Srinivas26k/srvdb/master/benchmarks/universal_benchmark.py
python universal_benchmark.py
```

## Interpreting Results

The script prints, for each mode:

- **Ingestion** (vectors/s): higher is better.
- **Search P50 / P99** (ms): typical and worst-1% query latency; lower is better.
- **Recall@10**: share of the exact top 10 (from a flat scan) that the mode
  returned; 100% means identical to exact search.
- **Memory**: process memory change; OS caching makes this noisy.

## Reporting results

SrvDB does not publish a number unless it can be reproduced. When you share
results, include:

- hardware (CPU model, cores, RAM) and OS,
- SrvDB version or commit hash,
- dataset (size, dimension, how it was generated),
- the exact command you ran.
