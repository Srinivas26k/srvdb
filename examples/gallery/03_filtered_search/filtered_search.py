"""Filtered search: recall and latency as the filter gets more selective.

Every record gets a ``bucket`` field from 0 to 99, so the filter
``{"bucket": {"$lt": s}}`` keeps about ``s`` percent of the data. For each
selectivity the script runs ``db.search(..., where=...)`` on an HNSW
database and compares the results with exact search over only the
matching records. The chart plots recall@10 and mean latency against the
fraction of records that match.

Run::

    python examples/gallery/03_filtered_search/filtered_search.py [--fast]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

DIM = 128
K = 10
PERCENTS = (1, 2, 5, 10, 25, 50, 100)


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("03 filtered search")
    n, nq = (3000, 40) if args.fast else (30000, 200)
    data, _ = make_blobs(n + nq, n_features=DIM, centers=40, random_state=1)
    data = data.astype(np.float32)
    data, queries = data[:n], data[n:]
    buckets = np.random.default_rng(1).integers(0, 100, size=n)
    print(f"dataset: make_blobs, {n} vectors, {nq} queries, dim {DIM}\n")

    rows = []
    with _common.scratch_dir() as tmp:
        with srvdb.create(str(tmp / "db"), dimension=DIM, index="hnsw") as db:
            db.add(
                [str(i) for i in range(n)],
                data.tolist(),
                [_common.meta(bucket=int(b)) for b in buckets],
            )
            for pct in PERCENTS:
                keep = np.flatnonzero(buckets < pct)
                truth = _common.exact_top_k(data[keep], queries, K)
                expected = [[str(keep[j]) for j in row] for row in truth]
                where = {"bucket": {"$lt": pct}}
                found = []
                with _common.Timer() as t:
                    for q in queries.tolist():
                        hits = db.search(q, k=K, where=where)
                        found.append([id_ for id_, _ in hits])
                recall = _common.recall_at_k(found, expected)
                ms = 1000 * t.seconds / nq
                share = len(keep) / n
                rows.append((share, recall, ms))
                print(
                    f"filter bucket<{pct:<3} matches {share:6.1%}  "
                    f"recall@{K}={recall:.3f}  latency={ms:.3f} ms"
                )

    plt = _common.pyplot()
    fig, ax = plt.subplots()
    shares = [r[0] for r in rows]
    ax.plot(shares, [r[1] for r in rows], marker="o", label=f"recall@{K}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("fraction of records matching the filter (log scale)")
    ax.set_ylabel(f"recall@{K} vs exact filtered search")
    twin = ax.twinx()
    twin.plot(
        shares,
        [r[2] for r in rows],
        marker="s",
        ls="--",
        color="#888888",
        label="mean latency (ms)",
    )
    twin.set_ylabel("mean latency per query (ms)")
    twin.grid(False)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right")
    ax.set_title(f"Filtered HNSW search, {n} vectors (this machine)")
    _common.save_chart(fig, args.out, "03_filtered_search.png")


if __name__ == "__main__":
    main()
