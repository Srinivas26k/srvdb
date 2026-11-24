"""Memory budget: watch ``index="auto"`` step down from flat to RaBitQ.

Creates ``srvdb.create(path, dim, index="auto", memory_mb=...)`` and adds
batches until the budget is full and ``add`` raises ``ValueError``. After
each batch it records ``db.memory_report()`` (mode, bits, estimated bytes)
and recall@10 against exact search. The chart has two panels sharing the
record count: estimated memory with the budget's 80 % limit, and recall,
with the mode changes marked.

Run::

    python examples/gallery/10_memory_budget/memory_budget.py [--fast]
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


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("10 memory budget")
    budget_mb, batch, nq = (1, 1000, 30) if args.fast else (8, 5000, 100)
    pool, _ = make_blobs(
        400_000 if not args.fast else 40_000,
        n_features=DIM,
        centers=100,
        random_state=10,
    )
    pool = pool.astype(np.float32)
    queries, pool = pool[:nq], pool[nq:]
    print(f"budget: memory_mb={budget_mb}, batches of {batch}, dim {DIM}\n")

    rows = []
    with _common.scratch_dir() as tmp:
        path = str(tmp / "budget")
        with srvdb.create(path, dimension=DIM, index="auto", memory_mb=budget_mb) as db:
            limit = db.memory_report()["limit_bytes"]
            stored = 0
            while stored + batch <= len(pool):
                chunk = pool[stored : stored + batch]
                try:
                    db.add(
                        [str(stored + i) for i in range(batch)],
                        chunk.tolist(),
                        ["{}"] * batch,
                    )
                except ValueError as err:
                    print(f"add #{stored // batch + 1} refused: {err}")
                    break
                stored += batch
                truth = _common.exact_top_k(pool[:stored], queries, K)
                found = [[i for i, _ in db.search(q, k=K)] for q in queries.tolist()]
                recall = _common.recall_at_k(
                    found, [[str(j) for j in r] for r in truth]
                )
                rep = db.memory_report()
                label = rep["mode"] + (f"{rep['bits']}" if rep["bits"] else "")
                changed = not rows or rows[-1][3] != label
                rows.append((stored, rep["total_bytes"], recall, label))
                if changed or len(rows) % 5 == 0:
                    print(
                        f"{stored:7d} records  {label:<8} "
                        f"{rep['total_bytes'] / 2**20:6.2f} MiB  "
                        f"recall@{K}={recall:.3f}"
                    )
            print(f"\nstored {db.count()} records; count after refusal is unchanged")
        with srvdb.open(path) as db:
            print(
                f"reopened in mode {db.memory_report()['mode']}, "
                f"bits {db.memory_report()['bits']}"
            )

    plt = _common.pyplot()
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5.5))
    counts = [r[0] for r in rows]
    a1.plot(counts, [r[1] / 2**20 for r in rows], marker=".")
    a1.set_ylim(bottom=0)
    a1.axhline(limit / 2**20, color="#888888", ls="--", lw=1, label="80 % of memory_mb")
    a1.set_ylabel("estimated memory (MiB)")
    a1.legend(loc="lower right")
    a2.plot(counts, [r[2] for r in rows], marker=".")
    a2.set_ylim(0, 1.05)
    a2.set_ylabel(f"recall@{K}")
    a2.set_xlabel("records stored")
    for prev, cur in zip(rows, rows[1:]):
        if cur[3] != prev[3]:
            for ax in (a1, a2):
                ax.axvline(cur[0], color="#bbbbbb", lw=1, ls=":")
            a2.annotate(cur[3], (cur[0], 0.05), fontsize=8, rotation=90)
    a1.set_title(f"index='auto', memory_mb={budget_mb} (this run)")
    _common.save_chart(fig, args.out, "10_memory_budget.png")


if __name__ == "__main__":
    main()
