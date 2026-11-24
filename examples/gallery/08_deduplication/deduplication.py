"""Deduplication: precision and recall of a cosine threshold.

Takes the digits images and plants near-duplicates: copies of 10 percent
of them with small pixel noise. After storing everything, each record's
nearest other record comes from ``db.search(vector, k=2)``; a pair is
flagged as a duplicate when its cosine similarity is at least the
threshold. Against the planted pairs the script reports precision and
recall for a sweep of thresholds and the best F1. The chart plots
precision and recall against the threshold.

Distinct handwritten digits of the same class can be very similar, so a
loose threshold flags them too; that is what the precision curve shows.

Run::

    python examples/gallery/08_deduplication/deduplication.py [--fast]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

THRESHOLDS = np.round(np.arange(0.90, 1.0001, 0.005), 3)


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("08 deduplication")
    rng = np.random.default_rng(8)
    base = load_digits().data
    n = 500 if args.fast else len(base)
    base = base[:n]
    src = rng.choice(n, size=n // 10, replace=False)
    dups = np.clip(base[src] + rng.normal(0, 0.6, size=base[src].shape), 0, 16)
    vectors = np.vstack([base, dups]).astype(np.float32)
    truth = {frozenset((int(s), n + j)) for j, s in enumerate(src)}
    print(f"{n} images plus {len(src)} planted near-duplicates\n")

    best_other = {}
    with _common.scratch_dir() as tmp:
        with srvdb.create(str(tmp / "dedup"), dimension=64, index="flat") as db:
            db.add(
                [str(i) for i in range(len(vectors))],
                vectors.tolist(),
                ["{}"] * len(vectors),
            )
            with _common.Timer() as t:
                for i, row in enumerate(vectors.tolist()):
                    for id_, score in db.search(row, k=2):
                        if int(id_) != i:
                            best_other[i] = (int(id_), score)
                            break
    print(f"nearest-neighbour pass: {t.seconds:.2f} s for {len(vectors)} records\n")

    rows = []
    for th in THRESHOLDS:
        pred = {frozenset((i, j)) for i, (j, s) in best_other.items() if s >= th}
        tp = len(pred & truth)
        precision = tp / len(pred) if pred else 1.0
        recall = tp / len(truth)
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        rows.append((float(th), precision, recall, f1))
        print(
            f"threshold {th:.3f}: flagged {len(pred):4d}  "
            f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}"
        )
    best = max(rows, key=lambda r: r[3])
    print(f"\nbest F1 {best[3]:.3f} at threshold {best[0]:.3f}")

    plt = _common.pyplot()
    fig, ax = plt.subplots()
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label="precision")
    ax.plot(
        [r[0] for r in rows], [r[2] for r in rows], marker="s", ls="--", label="recall"
    )
    ax.axvline(best[0], color="#999999", lw=1, ls=":", label=f"best F1 ({best[0]:.3f})")
    ax.set_xlabel("cosine similarity threshold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Near-duplicate detection, {len(vectors)} images (this run)")
    _common.save_chart(fig, args.out, "08_deduplication.png")


if __name__ == "__main__":
    main()
