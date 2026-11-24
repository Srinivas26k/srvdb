"""One adapter per library and index type.

Every adapter receives L2-normalised float32 vectors, so inner product equals
cosine similarity, which is SrvDB's metric and the ground truth's metric.

An adapter is a class with:
  build(base, threads, workdir)  -> None   (timed by the caller)
  set_param(value)               -> None   (one point of the sweep)
  search(query, k)               -> list of dataset row indices
  search_batch(queries, k)       -> list of lists of row indices
  sweep                          -> list of parameter values (or [None])
  param_name                     -> name of the swept parameter (or None)
"""

from __future__ import annotations

import os

import numpy as np

# Shared HNSW construction parameters (same for every HNSW library).
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
EF_SWEEP = [10, 16, 32, 64, 128, 256]
NPROBE_SWEEP = [1, 2, 4, 8, 16, 32]


def nlist_for(n: int) -> int:
    """Same default as the Rust harness: sqrt(n) partitions."""
    return max(1, int(n**0.5))


class Adapter:
    sweep: list = [None]
    param_name: str | None = None

    def set_param(self, value) -> None:
        pass

    def search_batch(self, queries: np.ndarray, k: int) -> list[list[int]]:
        return [self.search(q, k) for q in queries]


# ---------------------------------------------------------------- SrvDB


class SrvDB(Adapter):
    """SrvDB through its Python package (the API users call)."""

    def __init__(self, mode: str):
        self.mode = mode
        if mode == "hnsw":
            self.sweep, self.param_name = EF_SWEEP, "ef_search"

    def build(self, base, threads, workdir):
        import srvdb

        dim, path = base.shape[1], os.path.join(workdir, f"srvdb-{self.mode}")
        if self.mode == "hnsw":
            db = srvdb.Database.new_hnsw(path, dim, HNSW_M, HNSW_EF_CONSTRUCTION, 64)
        elif self.mode == "sq8":
            train = base[: min(len(base), 10_000)].tolist()
            db = srvdb.Database.new_scalar_quantized(path, dim, train)
        else:
            db = srvdb.Database(path, dim, "flat")
        for start in range(0, len(base), 1000):
            chunk = base[start : start + 1000]
            ids = [str(i) for i in range(start, start + len(chunk))]
            db.add(ids, chunk.tolist(), ["{}"] * len(chunk))
        if self.mode == "ivf":
            db.configure_ivf(nlist_for(len(base)), 8)
            db.set_mode("ivf")
            db.train_ivf()
        db.persist()
        self.db = db

    def set_param(self, value):
        if value is not None:
            self.db.set_ef_search(value)

    def search(self, q, k):
        return [int(i) for i, _ in self.db.search(q.tolist(), k)]

    def search_batch(self, queries, k):
        res = self.db.search_batch(queries.tolist(), k)
        return [[int(i) for i, _ in r] for r in res]


# ---------------------------------------------------------------- FAISS


class Faiss(Adapter):
    def __init__(self, kind: str):
        self.kind = kind
        if kind == "hnsw":
            self.sweep, self.param_name = EF_SWEEP, "ef_search"
        elif kind == "ivf":
            self.sweep, self.param_name = NPROBE_SWEEP, "nprobe"

    def build(self, base, threads, workdir):
        import faiss

        faiss.omp_set_num_threads(threads)
        d, ip = base.shape[1], faiss.METRIC_INNER_PRODUCT
        if self.kind == "flat":
            idx = faiss.IndexFlatIP(d)
        elif self.kind == "hnsw":
            idx = faiss.IndexHNSWFlat(d, HNSW_M, ip)
            idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        elif self.kind == "ivf":
            idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist_for(len(base)), ip)
            idx.train(base)
        else:
            idx = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, ip)
            idx.train(base[: min(len(base), 10_000)])
        idx.add(base)
        self.idx = idx

    def set_param(self, value):
        if self.kind == "hnsw":
            self.idx.hnsw.efSearch = value
        elif self.kind == "ivf":
            self.idx.nprobe = value

    def search(self, q, k):
        _, ids = self.idx.search(q.reshape(1, -1), k)
        return [int(i) for i in ids[0] if i >= 0]

    def search_batch(self, queries, k):
        _, ids = self.idx.search(queries, k)
        return [[int(i) for i in row if i >= 0] for row in ids]


# ---------------------------------------------------------------- hnswlib


class Hnswlib(Adapter):
    sweep, param_name = EF_SWEEP, "ef_search"

    def build(self, base, threads, workdir):
        import hnswlib

        self.threads = threads
        idx = hnswlib.Index(space="ip", dim=base.shape[1])
        idx.init_index(len(base), M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION)
        idx.add_items(base, np.arange(len(base)), num_threads=threads)
        self.idx = idx

    def set_param(self, value):
        self.idx.set_ef(value)

    def search(self, q, k):
        ids, _ = self.idx.knn_query(q.reshape(1, -1), k=k, num_threads=1)
        return [int(i) for i in ids[0]]

    def search_batch(self, queries, k):
        ids, _ = self.idx.knn_query(queries, k=k, num_threads=self.threads)
        return [[int(i) for i in row] for row in ids]


# ---------------------------------------------------------------- usearch


class Usearch(Adapter):
    sweep, param_name = EF_SWEEP, "ef_search"

    def build(self, base, threads, workdir):
        from usearch.index import Index

        self.threads = threads
        idx = Index(
            ndim=base.shape[1],
            metric="ip",
            dtype="f32",
            connectivity=HNSW_M,
            expansion_add=HNSW_EF_CONSTRUCTION,
        )
        idx.add(np.arange(len(base)), base, threads=threads)
        self.idx = idx

    def set_param(self, value):
        self.idx.expansion_search = value

    def search(self, q, k):
        return [int(i) for i in self.idx.search(q, k, threads=1).keys]

    def search_batch(self, queries, k):
        res = self.idx.search(queries, k, threads=self.threads)
        return [[int(i) for i in row] for row in res.keys]


# ---------------------------------------------------------------- turbovec


class Turbovec(Adapter):
    """turbovec TurboQuantIndex: quantised exhaustive scan; one point per bit width."""

    def __init__(self, bits: int):
        self.bits = bits

    def build(self, base, threads, workdir):
        import turbovec

        idx = turbovec.TurboQuantIndex(base.shape[1], self.bits)
        # Calibration needs a uniform random sample (about 1024 rows).
        rng = np.random.default_rng(0)
        sample = base[rng.choice(len(base), min(len(base), 1024), replace=False)]
        idx.calibrate(np.ascontiguousarray(sample))
        idx.add(base)
        idx.prepare()
        self.idx = idx

    def search(self, q, k):
        _, ids = self.idx.search(q.reshape(1, -1), k)
        return [int(i) for i in ids[0]]

    def search_batch(self, queries, k):
        _, ids = self.idx.search(queries, k)
        return [[int(i) for i in row] for row in ids]
