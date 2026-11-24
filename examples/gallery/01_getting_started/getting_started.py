"""Getting started: create, add, search, save, reopen and delete.

The smallest complete SrvDB program. It embeds the scikit-learn digits
images (8x8 pixels, 64 numbers each) as vectors, searches for the images
most like a held-out one, filters by label, and reopens the database to
show that saved data survives. No chart; everything printed is computed
by this run.

Run::

    python examples/gallery/01_getting_started/getting_started.py [--fast]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.datasets import load_digits  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("01 getting started")
    digits = load_digits()
    n = 300 if args.fast else len(digits.data)
    vectors = digits.data[:n].astype("float32")
    labels = digits.target[:n]
    query, query_label = vectors[-1], int(labels[-1])
    vectors, labels = vectors[:-1], labels[:-1]

    with _common.scratch_dir() as tmp:
        path = str(tmp / "digits-db")
        # 1. Create: the index and dimension are recorded on disk.
        with srvdb.create(path, dimension=64, index="flat") as db:
            ids = [f"img{i}" for i in range(len(vectors))]
            metas = [_common.meta(label=int(y)) for y in labels]
            db.add(ids, vectors.tolist(), metas)
            print(f"stored {db.count()} images: {db.info()}")
        # Leaving the `with` block saved and closed the database.

        # 2. Reopen with its recorded settings and search.
        with srvdb.open(path) as db:
            print(f"\nquery: a held-out image of the digit {query_label}")
            for rank, (id_, score) in enumerate(db.search(query.tolist(), k=5), 1):
                label = json.loads(db.get(id_))["label"]
                print(f"  {rank}. {id_:<8} label={label} cosine={score:.4f}")

            # 3. Filtered search runs inside the database.
            other = (query_label + 1) % 10
            hits = db.search(query.tolist(), k=3, where={"label": other})
            print(f"\nclosest images labelled {other}:")
            for id_, score in hits:
                print(f"  {id_:<8} cosine={score:.4f}")

            # 4. Classify the query by a vote of its 5 nearest neighbours.
            votes = [
                json.loads(db.get(i))["label"]
                for i, _ in db.search(query.tolist(), k=5)
            ]
            guess = max(set(votes), key=votes.count)
            print(f"\n5-NN vote: {guess} (true label {query_label})")

        # 5. Delete the closed database directory.
        srvdb.delete(path)
        print(f"deleted: {not Path(path).exists()}")


if __name__ == "__main__":
    main()
