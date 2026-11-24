"""Measure one library/index in its own process (so peak RSS is per index).

Invoked by run.py; prints one JSON object on the last stdout line.
"""

from __future__ import annotations

import json
import resource
import sys
import tempfile
import time

import numpy as np

import adapters
from data import load_dataset

# name -> (library label, index label, factory)
REGISTRY = {
    "srvdb-flat": ("SrvDB", "flat", lambda: adapters.SrvDB("flat")),
    "srvdb-hnsw": ("SrvDB", "hnsw", lambda: adapters.SrvDB("hnsw")),
    "srvdb-sq8": ("SrvDB", "sq8", lambda: adapters.SrvDB("sq8")),
    "srvdb-ivf": ("SrvDB", "ivf", lambda: adapters.SrvDB("ivf")),
    "faiss-flat": ("FAISS", "IndexFlatIP", lambda: adapters.Faiss("flat")),
    "faiss-hnsw": ("FAISS", "IndexHNSWFlat", lambda: adapters.Faiss("hnsw")),
    "faiss-ivf": ("FAISS", "IndexIVFFlat", lambda: adapters.Faiss("ivf")),
    "faiss-sq8": ("FAISS", "IndexScalarQuantizer", lambda: adapters.Faiss("sq8")),
    "hnswlib": ("hnswlib", "hnsw", adapters.Hnswlib),
    "usearch": ("usearch", "hnsw", adapters.Usearch),
    "turbovec-4bit": ("turbovec", "TurboQuant 4-bit", lambda: adapters.Turbovec(4)),
    "turbovec-2bit": ("turbovec", "TurboQuant 2-bit", lambda: adapters.Turbovec(2)),
}

WARMUP = 10


def rss_mib() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def recall(truth: np.ndarray, found: list[list[int]], k: int) -> float:
    hits = sum(len(set(t[:k]) & set(f[:k])) for t, f in zip(truth, found))
    return hits / (len(truth) * k)


def measure_point(ad, queries, truth, k) -> dict:
    for q in queries[:WARMUP]:
        ad.search(q, k)
    lat, found = [], []
    t0 = time.perf_counter()
    for q in queries:
        s = time.perf_counter()
        found.append(ad.search(q, k))
        lat.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    lat.sort()

    def pct(p: float) -> float:
        i = max(0, min(len(lat) - 1, int(np.ceil(p / 100 * len(lat))) - 1))
        return lat[i] * 1000

    t1 = time.perf_counter()
    batch_found = ad.search_batch(queries, k)
    batch_secs = time.perf_counter() - t1
    return {
        "recall_at_k": recall(truth, found, k),
        "batch_recall_at_k": recall(truth, batch_found, k),
        "qps": len(queries) / total,
        "p50_ms": pct(50),
        "p99_ms": pct(99),
        "batch_qps": len(queries) / batch_secs,
    }


def main() -> None:
    name, data_dir, threads, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    library, index, factory = REGISTRY[name]
    base, queries, truth = load_dataset(data_dir)
    baseline = rss_mib()
    ad = factory()
    with tempfile.TemporaryDirectory(dir=sys.argv[5] if len(sys.argv) > 5 else None) as wd:
        t0 = time.perf_counter()
        ad.build(base, threads, wd)
        build_secs = time.perf_counter() - t0
        points = []
        for value in ad.sweep:
            ad.set_param(value)
            p = measure_point(ad, queries, truth, k)
            p["param"], p["value"] = ad.param_name, value
            points.append(p)
        peak = rss_mib()
    print(json.dumps({
        "name": name, "library": library, "index": index, "threads": threads,
        "build_secs": build_secs, "baseline_rss_mib": baseline,
        "peak_rss_mib": peak, "points": points,
    }))


if __name__ == "__main__":
    main()
