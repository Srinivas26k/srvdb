"""Competitor quantizers on the same cached datasets as run.py.

    .competitor-venv/bin/python benchmarks/social/competitors.py [datasets]

Runs in its own venv (faiss-cpu, turbovec; never SrvDB dependencies), on
the base/query/ground-truth arrays run.py cached under .data/cache/, and
merges results into results.json under datasets.<name>.competitors.
Measures recall@10 and code bytes per vector only; no speed.

Code bytes: FAISS `sa_code_size()` (the per-vector code, including RaBitQ's
per-vector factors); turbovec `len(to_bytes()) / n` (serialized index,
so a small fixed header is included). Rerank variants also keep float32
vectors (4*dim bytes) outside the code, as SrvDB's RaBitQ does.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".data" / "cache"
OUT = HERE / "results.json"
K = 10
RERANK = {4: 10, 2: 20, 1: 40}  # SrvDB's default rerank factors per bit width


def recall(ids: np.ndarray, gt: np.ndarray) -> float:
    return float(
        np.mean([len(set(a) & set(b)) / K for a, b in zip(ids.tolist(), gt.tolist())])
    )


def rerank(base, q, cand: np.ndarray) -> np.ndarray:
    """Exact cosine rerank of candidate ids (inputs L2-normalised)."""
    out = np.empty((len(q), K), dtype=np.int64)
    for i, c in enumerate(cand):
        c = c[c >= 0]
        s = base[c] @ q[i]
        out[i] = c[np.argsort(-s)[:K]]
    return out


def faiss_runs(base, q, gt, train):
    import faiss

    d = base.shape[1]
    ip = faiss.METRIC_INNER_PRODUCT
    rows = []

    def run(name, index, extra=None, k_factor=None, note=""):
        if not index.is_trained:
            index.train(train)
        index.add(base)
        code = index.sa_code_size()
        # Refine = take k * k_factor candidates, rescore exactly in float32
        # (what IndexRefineFlat does), shared with the binary baseline.
        _, ids = index.search(q, K * (k_factor or 1))
        if k_factor:
            ids = rerank(base, q, ids)
        rows.append(
            {
                "library": "faiss",
                "index": name,
                "code_bytes_per_vector": float(code),
                "recall_at_10": round(recall(ids, gt), 4),
                "rerank_k_factor": k_factor,
                "params": extra or {},
                "note": note,
            }
        )

    for bits in (4, 2, 1):
        for kf in (None, RERANK[bits]):
            run(
                f"IndexRaBitQ {bits}-bit" + (" + refine" if kf else ""),
                faiss.IndexRaBitQ(d, ip, bits),
                {"nb_bits": bits},
                kf,
            )
    run(
        "IndexScalarQuantizer QT_8bit",
        faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, ip),
    )
    m = max(x for x in range(1, d // 8 + 1) if d % x == 0)
    run(
        f"IndexPQ M={m} x 8-bit",
        faiss.IndexPQ(d, m, 8, ip),
        {"M": m},
        note="M = largest divisor of dim <= dim/8",
    )
    if d <= 256:
        opq = faiss.IndexPreTransform(faiss.OPQMatrix(d, m), faiss.IndexPQ(d, m, 8, ip))
        run(f"OPQ + PQ M={m} x 8-bit", opq, {"M": m})
    # pgvector-style: sign bits, Hamming scan, optional float rerank.
    bits = np.packbits(base > 0, axis=1)
    qbits = np.packbits(q > 0, axis=1)
    b = faiss.IndexBinaryFlat(bits.shape[1] * 8)  # sign bits padded to bytes
    b.add(bits)
    for kf in (None, 40):
        _, cand = b.search(qbits, K * (kf or 1))
        ids = rerank(base, q, cand) if kf else cand
        rows.append(
            {
                "library": "faiss",
                "index": "IndexBinaryFlat sign bits" + (" + rerank" if kf else ""),
                "code_bytes_per_vector": float(b.code_size),
                "recall_at_10": round(recall(ids, gt), 4),
                "rerank_k_factor": kf,
                "params": {},
                "note": "pgvector-style bit + float rerank",
            }
        )
    return rows


def turbovec_runs(base, q, gt):
    import turbovec

    rows = []
    if base.shape[1] % 8:
        # turbovec rejects it: "dim must be a positive multiple of 8".
        return [
            {
                "library": "turbovec",
                "index": "TurboQuantIndex",
                "skipped": f"dim {base.shape[1]} not a multiple of 8",
            }
        ]
    for bits in (4, 2):
        idx = turbovec.TurboQuantIndex(base.shape[1], bits)
        rng = np.random.default_rng(0)
        idx.calibrate(
            np.ascontiguousarray(base[rng.choice(len(base), 1024, replace=False)])
        )
        idx.add(base)
        idx.prepare()
        _, ids = idx.search(q, K)
        rows.append(
            {
                "library": "turbovec",
                "index": f"TurboQuantIndex {bits}-bit",
                "code_bytes_per_vector": round(len(idx.to_bytes()) / len(base), 2),
                "recall_at_10": round(recall(np.asarray(ids), gt), 4),
                "rerank_k_factor": None,
                "params": {"bits": bits, "calibration_rows": 1024},
                "note": "bytes = serialized index / n (includes header)",
            }
        )
    return rows


def main() -> None:
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    res = json.loads(OUT.read_text())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=HERE,
    ).stdout.strip()
    res["competitor_run"] = {
        "git_commit": commit,
        "command": " ".join(
            [
                ".competitor-venv/bin/python",
                "benchmarks/social/competitors.py",
                *sys.argv[1:],
            ]
        ),
    }
    res["competitor_versions"] = {
        p: {
            "version": md.version(p),
            "licence": md.metadata(p).get("License-Expression")
            or md.metadata(p).get("License"),
        }
        for p in ("faiss-cpu", "turbovec", "numpy")
    }
    for name in names or list(res["datasets"]):
        src = CACHE / name
        base = np.ascontiguousarray(np.load(src / "base.npy"))
        q, gt = np.load(src / "queries.npy"), np.load(src / "gt.npy")
        rng = np.random.default_rng(0)
        train = base[rng.choice(len(base), min(len(base), 50_000), replace=False)]
        rows = faiss_runs(base, q, gt, train) + turbovec_runs(base, q, gt)
        res["datasets"][name]["competitors"] = rows
        OUT.write_text(json.dumps(res, indent=1) + "\n")
        for r in [r for r in rows if "skipped" not in r]:
            print(
                f"{name} {r['library']} {r['index']}: recall@10 {r['recall_at_10']}, "
                f"{r['code_bytes_per_vector']} B/vec",
                flush=True,
            )
        del base


if __name__ == "__main__":
    main()
