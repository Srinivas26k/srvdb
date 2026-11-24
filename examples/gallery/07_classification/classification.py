"""Classification: SrvDB k-NN against scikit-learn's KNeighborsClassifier.

Splits the digits dataset 75/25, stores the training images in SrvDB
(flat and HNSW) and labels each test image by majority vote of its k
nearest stored neighbours, read from metadata. The same vote with
``KNeighborsClassifier(metric="cosine", algorithm="brute")`` is the
reference. Prints accuracy and mean latency per test image for each, and
how often SrvDB and scikit-learn predict the same label; the chart shows
accuracy and latency side by side.

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


def srvdb_knn(path: str, index: str, xtr, ytr, xte) -> tuple[list[int], float]:
    with srvdb.create(path, dimension=64, index=index) as db:
        db.add(
            [str(i) for i in range(len(xtr))],
            xtr.tolist(),
            [_common.meta(label=int(y)) for y in ytr],
        )
        # Labels are also in the metadata (db.get(id)); a dict avoids a
        # JSON parse per neighbour in the timed loop.
        label_of = {str(i): int(y) for i, y in enumerate(ytr)}
        preds = []
        with _common.Timer() as t:
            for row in xte.tolist():
                votes = Counter(label_of[id_] for id_, _ in db.search(row, k=K))
                preds.append(votes.most_common(1)[0][0])
    return preds, 1000 * t.seconds / len(xte)


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

    clf = KNeighborsClassifier(K, metric="cosine", algorithm="brute").fit(xtr, ytr)
    with _common.Timer() as t:
        sk = [int(clf.predict(row[None, :])[0]) for row in xte]
    results = {"scikit-learn": (sk, 1000 * t.seconds / len(xte))}
    with _common.scratch_dir() as tmp:
        for index in ("flat", "hnsw"):
            results[f"srvdb {index}"] = srvdb_knn(
                str(tmp / index), index, xtr, ytr, xte
            )

    rows = []
    for name, (preds, ms) in results.items():
        acc = float(np.mean(np.array(preds) == yte))
        agree = float(np.mean(np.array(preds) == np.array(sk)))
        rows.append((name, acc, ms))
        print(
            f"{name:<13} accuracy={acc:.4f}  latency={ms:.3f} ms/image  "
            f"agrees with scikit-learn on {agree:.1%}"
        )

    plt = _common.pyplot()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    names = [r[0] for r in rows]
    a1.bar(names, [r[1] for r in rows], color="#222222", width=0.55)
    a1.set_ylim(min(r[1] for r in rows) - 0.05, 1.0)
    a1.set_title(f"test accuracy ({K}-NN)")
    a2.bar(names, [r[2] for r in rows], color="#777777", width=0.55)
    a2.set_title("mean latency per image (ms)")
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Digits k-NN, {len(xtr)} train / {len(xte)} test (this machine)")
    _common.save_chart(fig, args.out, "07_classification.png")


if __name__ == "__main__":
    main()
