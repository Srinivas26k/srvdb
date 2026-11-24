"""RaBitQ mode through the Python package (ADR 0008)."""

import json
import math
import random

import pytest

import srvdb

DIM = 128
N = 800
K = 10


def clustered(n, seed=0, clusters=20):
    rng = random.Random(seed)
    centres = [[rng.random() - 0.5 for _ in range(DIM)] for _ in range(clusters)]
    return [
        [c + 0.6 * (rng.random() - 0.5) for c in centres[i % clusters]]
        for i in range(n)
    ]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def fill(db, vectors):
    ids = [f"doc{i}" for i in range(len(vectors))]
    db.add(ids, vectors, [json.dumps({"i": i}) for i in range(len(vectors))])
    db.persist()


def recall(db, vectors, queries):
    hits = 0
    for q in queries:
        exact = sorted(range(len(vectors)), key=lambda i: -cosine(q, vectors[i]))[:K]
        truth = {f"doc{i}" for i in exact}
        hits += sum(doc in truth for doc, _ in db.search(q, K))
    return hits / (len(queries) * K)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_recall_with_rerank_and_reopen(tmp_path, bits):
    vectors = clustered(N)
    queries = clustered(15, seed=1)
    db = srvdb.Database.new_rabitq(str(tmp_path), dimension=DIM, bits=bits)
    fill(db, vectors)
    assert recall(db, vectors, queries) >= 0.9
    db.set_rabitq_rerank(0)
    assert recall(db, vectors, queries) > 0.2
    del db

    db = srvdb.Database(str(tmp_path), dimension=DIM, mode=f"rabitq{bits}")
    assert db.count() == N
    assert db.search(vectors[3], 1)[0][0] == "doc3"


def test_bad_settings_raise_value_error(tmp_path):
    for bits in (0, 3, 8):
        with pytest.raises(ValueError):
            srvdb.Database.new_rabitq(
                str(tmp_path / str(bits)), dimension=DIM, bits=bits
            )
    with pytest.raises(ValueError):
        srvdb.Database(str(tmp_path / "m"), dimension=DIM, mode="rabitq3")
    flat = srvdb.Database(str(tmp_path / "flat"), dimension=DIM)
    with pytest.raises(ValueError):
        flat.set_rabitq_rerank(5)
