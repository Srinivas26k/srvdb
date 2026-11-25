"""Classification: SrvDB k-NN against scikit-learn's KNeighborsClassifier.

Splits the digits dataset 75/25, stores the training images in SrvDB
(flat and HNSW) and labels each test image by majority vote of its k
nearest stored neighbours. The same vote with
``KNeighborsClassifier(metric="cosine", algorithm="brute")`` is the
reference. Prints accuracy, how often SrvDB and scikit-learn agree, and
latency per test image measured two ways for every classifier:

- per call: one image at a time (``db.search`` / ``predict`` on one row),
  which includes each library's per-call Python overhead;
- batch: the whole test set in one call (``db.search_batch`` /
  ``predict`` on all rows), divided by the number of images.

The chart shows accuracy (bars from zero, values labelled) and both
latencies side by side.

Run::

    python examples/gallery/07_classification/classification.py [--fast]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

K = 5


def vote(hits, label_of: dict[str, int]) -> int:
    return Counter(label_of[id_] for id_, _ in hits).most_common(1)[0][0]


def srvdb_knn(path: str, index: str, xtr, ytr, xte):
    """Predictions, per-call ms/image and batch ms/image."""
    with srvdb.create(path, dimension=64, index=index) as db:
        db.add(
            [str(i) for i in range(len(xtr))],
            xtr.tolist(),
            [_common.meta(label=int(y)) for y in ytr],
        )
        # Labels are also in the metadata (db.get(id)); a dict avoids a
        # JSON parse per neighbour in the timed loops.
        label_of = {str(i): int(y) for i, y in enumerate(ytr)}
        rows = xte.tolist()
        with _common.Timer() as single:
            preds = [vote(db.search(row, k=K), label_of) for row in rows]
        with _common.Timer() as batch:
            batched = [vote(h, label_of) for h in db.search_batch(rows, k=K)]
    same = np.mean(np.array(batched) == np.array(preds))
    print(
        f"srvdb {index}: search_batch and search predict the same label "
        f"for {same:.1%} of images"
    )
    return preds, single.seconds, batch.seconds


def sklearn_knn(xtr, ytr, xte):
    clf = KNeighborsClassifier(K, metric="cosine", algorithm="brute").fit(xtr, ytr)
    with _common.Timer() as single:
        preds = [int(clf.predict(row[None, :])[0]) for row in xte]
    with _common.Timer() as batch:
        clf.predict(xte)
    return preds, single.seconds, batch.seconds


def chart(rows, out: Path, title: str) -> None:
    plt = _common.pyplot()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    names = [r[0] for r in rows]
    x = np.arange(len(rows))
    _common.label_bars(
        a1, a1.bar(names, [r[1] for r in rows], color="#222222", width=0.55)
    )
    a1.set_ylim(0, 1.12)
    a1.set_title(f"test accuracy ({K}-NN)")
    per_call = a2.bar(
        x - 0.2,
        [r[2] for r in rows],
        0.4,
        color="#999999",
        label="per call (one image)",
    )
    batch = a2.bar(
        x + 0.2,
        [r[3] for r in rows],
        0.4,
        color="#222222",
        label="batch (all images, per image)",
    )
    _common.label_bars(a2, per_call, "{:.3f}")
    _common.label_bars(a2, batch, "{:.3f}")
    a2.set_xticks(x, names)
    a2.set_title("latency per image (ms)")
    a2.legend(loc="upper right", fontsize=8)
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
    fig.suptitle(title)
    _common.save_chart(fig, out, "07_classification.png")


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("07 classification")
    digits = load_digits()
    n = 600 if args.fast else len(digits.data)
    x = _common.normalize(digits.data[:n])
    xtr, xte, ytr, yte = train_test_split(
        x, digits.target[:n], test_size=0.25, random_state=0, stratify=digits.target[:n]
    )
    print(f"digits: {len(xtr)} training images, {len(xte)} test images, k={K}\n")

    results = {"scikit-learn": sklearn_knn(xtr, ytr, xte)}
    with _common.scratch_dir() as tmp:
        for index in ("flat", "hnsw"):
            results[f"srvdb {index}"] = srvdb_knn(
                str(tmp / index), index, xtr, ytr, xte
            )

    sk = np.array(results["scikit-learn"][0])
    rows = []
    for name, (preds, single, batch) in results.items():
        acc = float(np.mean(np.array(preds) == yte))
        agree = float(np.mean(np.array(preds) == sk))
        per_call, per_batch = 1000 * single / len(xte), 1000 * batch / len(xte)
        rows.append((name, acc, per_call, per_batch))
        print(
            f"{name:<13} accuracy={acc:.4f}  agrees with scikit-learn {agree:6.1%}  "
            f"per call {per_call:.4f} ms/image  batch {per_batch:.4f} ms/image"
        )
    print(
        "\nper call includes each library's Python call overhead; batch is the "
        "fairer comparison of the search itself"
    )
    chart(
        rows,
        args.out,
        f"Digits k-NN, {len(xtr)} train / {len(xte)} test (this machine)",
    )


if __name__ == "__main__":
    main()
