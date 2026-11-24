"""Index modes: recall, queries per second and memory for each index.

Builds the same clustered synthetic dataset (scikit-learn ``make_blobs``)
in flat, HNSW, IVF, SQ8, PQ and RaBitQ databases, then measures for each:
recall@10 against exact search, single-query throughput and the memory
estimate from ``db.memory_report()``. Prints a table and saves a
three-panel bar chart. Numbers depend on the machine; read the table, not
the chart from someone else's run.

Run::

    python examples/gallery/02_index_modes/index_modes.py [--fast]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

DIM = 128
K = 10
MODES = {
    "flat": {},
    "hnsw": {"ef_search": 64},
    "ivf": {"nlist": 64, "nprobe": 8},
    "sq8": {},
    "pq": {},
    "rabitq": {"bits": 4},
}


def build(path: str, mode: str, data: np.ndarray, options: dict) -> srvdb.Database:
    if mode in ("sq8", "pq"):
        options = {**options, "training": data[:2000].tolist()}
    db = srvdb.create(path, dimension=DIM, index=mode, **options)
    ids = [str(i) for i in range(len(data))]
    db.add(ids, data.tolist(), ["{}"] * len(data))
    if mode == "ivf":
        db.train_ivf()
    db.save()
    return db


def measure(db: srvdb.Database, queries: np.ndarray, truth: np.ndarray):
    found = []
    with _common.Timer() as t:
        for q in queries.tolist():
            found.append([id_ for id_, _ in db.search(q, k=K)])
    expected = [[str(i) for i in row] for row in truth]
    report = db.memory_report()
    return (
        _common.recall_at_k(found, expected),
        len(queries) / t.seconds,
        report["total_bytes"] / 2**20,
    )


def chart(rows: list[tuple], out: Path, n: int) -> None:
    plt = _common.pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    names = [r[0] for r in rows]
    for ax, col, title in zip(
        axes, (1, 2, 3), (f"recall@{K}", "queries / second", "memory estimate (MiB)")
    ):
        values = [r[col] for r in rows]
        ax.bar(names, values, color="#222222", width=0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.05)
    fig.suptitle(f"SrvDB index modes, {n} vectors x {DIM} dims (this machine)", y=1.04)
    _common.save_chart(fig, out, "02_index_modes.png")


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("02 index modes")
    n, nq = (3000, 50) if args.fast else (30000, 300)
    data, _ = make_blobs(n + nq, n_features=DIM, centers=50, random_state=0)
    data = data.astype(np.float32)
    data, queries = data[:n], data[n:]
    truth = _common.exact_top_k(data, queries, K)
    print(f"dataset: make_blobs, {n} vectors, {nq} queries, dim {DIM}\n")

    rows = []
    with _common.scratch_dir() as tmp:
        for mode, options in MODES.items():
            with _common.Timer() as t:
                db = build(str(tmp / mode), mode, data, options)
            with db:
                recall, qps, mib = measure(db, queries, truth)
            rows.append((mode, recall, qps, mib, t.seconds))
    table = pd.DataFrame(
        rows, columns=["mode", f"recall@{K}", "qps", "memory_mib", "build_s"]
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "02_index_modes.csv", index=False)
    print(f"table: {args.out / '02_index_modes.csv'}")
    chart(rows, args.out, n)


if __name__ == "__main__":
    main()
