"""Clustering and visualization: KMeans and PCA over stored vectors.

Stores the scikit-learn digits images in SrvDB, clusters the same vectors
with KMeans, and uses SrvDB to find each cluster's representative: the
stored image closest to the centroid (``db.search(centroid, k=1)``). It
prints the adjusted Rand index of the clusters against the true labels and
saves a PCA scatter plot of the vectors, each cluster a different grey and
marker, with the representatives circled.

SrvDB returns ids and scores, not raw vectors, so keep the array you
embedded (or re-embed) when you need it for clustering.

Run::

    python examples/gallery/06_clustering/clustering.py [--fast]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

MARKERS = "o^sDv<>pPX"


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("06 clustering and visualization")
    digits = load_digits()
    n = 400 if args.fast else len(digits.data)
    vectors = _common.normalize(digits.data[:n])
    labels = digits.target[:n]

    km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(vectors)
    ari = adjusted_rand_score(labels, km.labels_)
    print(f"{n} digit images, KMeans k=10, adjusted Rand index = {ari:.3f}\n")

    reps = []
    with _common.scratch_dir() as tmp:
        with srvdb.create(str(tmp / "digits"), dimension=64, index="flat") as db:
            db.add(
                [str(i) for i in range(n)],
                vectors.tolist(),
                [_common.meta(label=int(y)) for y in labels],
            )
            for c, centroid in enumerate(km.cluster_centers_):
                ((id_, score),) = db.search(centroid.tolist(), k=1)
                label = json.loads(db.get(id_))["label"]
                size = int(np.sum(km.labels_ == c))
                reps.append(int(id_))
                print(
                    f"cluster {c}: {size:4d} images, representative "
                    f"img{id_} (label {label}, cosine {score:.3f})"
                )

    xy = PCA(n_components=2, random_state=0).fit_transform(vectors)
    plt = _common.pyplot()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    greys = plt.get_cmap("Greys")
    for c in range(10):
        pts = xy[km.labels_ == c]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=10,
            marker=MARKERS[c],
            color=greys(0.35 + 0.6 * c / 9),
            label=f"cluster {c}",
        )
    ax.scatter(
        xy[reps, 0],
        xy[reps, 1],
        s=160,
        facecolors="none",
        edgecolors="#000000",
        linewidths=1.5,
        label="representative",
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"Digits: KMeans clusters on PCA axes (ARI {ari:.2f}, this run)")
    _common.save_chart(fig, args.out, "06_clustering.png")


if __name__ == "__main__":
    main()
