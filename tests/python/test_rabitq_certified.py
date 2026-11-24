"""Certified and deadline search through Python (ADR 0017)."""

import json
import math
import random

import pytest

import srvdb

DIM = 128
K = 10


def clustered(n, seed, clusters=20):
    rng = random.Random(seed)
    centres = [[rng.random() - 0.5 for _ in range(DIM)] for _ in range(clusters)]
    return [
        [c + 0.6 * (rng.random() - 0.5) for c in centres[i % clusters]]
        for i in range(n)
    ]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def rabitq_db(path, vectors, bits=4):
    db = srvdb.Database.new_rabitq(str(path), dimension=DIM, bits=bits)
    db.add(
        [f"doc{i}" for i in range(len(vectors))],
        vectors,
        [json.dumps({"i": i}) for i in range(len(vectors))],
    )
    return db


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_certified_recall_meets_confidence(tmp_path, seed):
    vectors = clustered(600, seed)
    queries = clustered(8, seed + 100)
    db = rabitq_db(tmp_path, vectors)
    hits = 0
    for q in queries:
        out = db.search(q, K, certify=True)
        assert len(out) == K
        assert out.completeness == 1.0
        exact = sorted(range(len(vectors)), key=lambda i: -cosine(q, vectors[i]))[:K]
        truth = {f"doc{i}" for i in exact}
        hits += sum(doc in truth for doc, _ in out)
    assert hits / (K * len(queries)) >= out.confidence


def test_deadline_zero_and_generous(tmp_path):
    vectors = clustered(800, 5)
    db = rabitq_db(tmp_path, vectors)
    q = vectors[3]
    assert db.search(q, K, deadline_ms=0).completeness < 1.0
    out = db.search(q, K, deadline_ms=60_000)
    assert out.completeness == 1.0
    assert out.confidence is None
    assert list(out) == db.search(q, K)
    assert out[0] == out.results[0]


def test_other_modes_and_bad_arguments_raise(tmp_path):
    db = srvdb.create(str(tmp_path / "flat"), dimension=DIM, index="flat")
    with pytest.raises(ValueError, match="RaBitQ"):
        db.search([0.1] * DIM, K, certify=True)
    with pytest.raises(ValueError, match="RaBitQ"):
        db.search([0.1] * DIM, K, deadline_ms=5)
    rq = rabitq_db(tmp_path / "rq", clustered(50, 1))
    with pytest.raises(ValueError):
        rq.search([0.1] * DIM, K, deadline_ms=-1)
    with pytest.raises(ValueError):
        rq.search([0.1] * DIM, K, certify=True, where={"i": 1})


def test_outcome_supports_slices(tmp_path):
    vectors = clustered(200, 9)
    db = rabitq_db(tmp_path, vectors)
    out = db.search(vectors[0], 5, certify=True)
    assert out[1:3] == out.results[1:3]
    assert out[-1] == out.results[-1]
    with pytest.raises(IndexError):
        out[99]
