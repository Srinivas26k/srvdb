"""Dataset: exported by the Rust harness so every library sees the same bytes.

`cargo run --release -p srvdb --example bench -- --export-fvecs DIR` writes
the harness's seeded synthetic dataset (SplitMix64 clustered Gaussians) as
base.fvecs and query.fvecs. Here the vectors are L2-normalised, so inner
product equals cosine (SrvDB's metric), and the exact top-k is computed by
brute force and cached as truth.npy.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1):
        raise ValueError(f"{path}: not a valid .fvecs file")
    rows = raw.reshape(-1, dim + 1)
    if np.any(rows[:, 0] != dim):
        raise ValueError(f"{path}: mixed dimensions")
    return rows[:, 1:].copy().view(np.float32)


def normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.where(n == 0, 1, n), dtype=np.float32)


def exact_topk(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), 256):
        sims = queries[s : s + 256] @ base.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        out[s : s + 256] = np.take_along_axis(part, order, axis=1)
    return out


def prepare(repo: str, root: str, dim: int, n: int, nq: int, seed: int, k: int):
    """Export (once) and return (data_dir, dataset description)."""
    d = os.path.join(root, f"synthetic-d{dim}-n{n}-q{nq}-s{seed}")
    meta_path = os.path.join(d, "dataset.json")
    if not os.path.exists(meta_path):
        cmd = [
            "cargo", "run", "--quiet", "--release", "-p", "srvdb", "--example", "bench",
            "--", "--dim", str(dim), "--n", str(n), "--queries", str(nq),
            "--seed", str(seed), "--export-fvecs", d,
        ]
        out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True)
        meta = json.loads(out.stdout.strip().splitlines()[-1])
        base = normalise(read_fvecs(os.path.join(d, "base.fvecs")))
        queries = normalise(read_fvecs(os.path.join(d, "query.fvecs")))
        np.save(os.path.join(d, "base.npy"), base)
        np.save(os.path.join(d, "queries.npy"), queries)
        np.save(os.path.join(d, "truth.npy"), exact_topk(base, queries, k))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return d, {
        "name": meta["name"], "dim": meta["dim"], "base": meta["base"],
        "queries": meta["queries"], "seed": meta["seed"], "k": k,
        "generator": "crates/srvdb/examples/bench/dataset.rs (exported with --export-fvecs)",
        "metric": "cosine (vectors L2-normalised; inner product for other libraries)",
    }


def load_dataset(d: str):
    return (
        np.load(os.path.join(d, "base.npy")),
        np.load(os.path.join(d, "queries.npy")),
        np.load(os.path.join(d, "truth.npy")),
    )
