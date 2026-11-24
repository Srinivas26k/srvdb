"""Filtered search from Python: ``db.search(query, k, where={...})`` (#46)."""

import json
import math

import numpy as np
import pytest

import srvdb

DIM = 32
N = 2000


def dataset(seed=7):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(20, DIM))
    labels = rng.integers(0, 20, size=N)
    vectors = centres[labels] + 0.6 * rng.normal(size=(N, DIM))
    metas = [
        {
            "bucket": int(i % 100),
            "lang": ["en", "de"][i % 2],
            "info": {"year": 1900 + i % 120},
        }
        for i in range(N)
    ]
    return vectors.astype(np.float32), metas


@pytest.fixture(scope="module")
def filled(tmp_path_factory):
    vectors, metas = dataset()
    db = srvdb.create(str(tmp_path_factory.mktemp("f")), DIM, "hnsw")
    db.add(
        [f"v{i}" for i in range(N)],
        vectors.tolist(),
        [json.dumps(m) for m in metas],
    )
    yield db, vectors, metas
    db.close()


def exact(vectors, metas, query, k, keep):
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = unit @ (query / np.linalg.norm(query))
    ids = [i for i in np.argsort(-scores) if keep(metas[i])]
    return [f"v{i}" for i in ids[:k]]


@pytest.mark.parametrize(
    ("where", "keep"),
    [
        ({"bucket": {"$lt": 1}}, lambda m: m["bucket"] < 1),
        ({"bucket": {"$lt": 10}}, lambda m: m["bucket"] < 10),
        (
            {"lang": "en", "info.year": {"$gte": 1960}},
            lambda m: m["lang"] == "en" and m["info"]["year"] >= 1960,
        ),
        (
            {"$or": [{"bucket": 3}, {"bucket": {"$in": [50, 51]}}]},
            lambda m: m["bucket"] in (3, 50, 51),
        ),
    ],
)
def test_recall_against_exact_filtered_search(filled, where, keep):
    db, vectors, metas = filled
    rng = np.random.default_rng(1)
    k, hits, total = 10, 0, 0
    for _ in range(20):
        query = vectors[rng.integers(N)] + 0.1 * rng.normal(size=DIM).astype(np.float32)
        got = db.search(query.tolist(), k, where=where)
        truth = exact(vectors, metas, query, k, keep)
        assert len(got) == len(truth) == k
        assert all(keep(metas[int(id_[1:])]) for id_, _ in got)
        hits += len({id_ for id_, _ in got} & set(truth))
        total += k
    assert hits / total >= 0.9


def test_fewer_matches_than_k_returns_every_match(filled):
    db, vectors, _ = filled
    got = db.search(vectors[0].tolist(), 50, where={"bucket": 7, "lang": "de"})
    assert sorted(id_ for id_, _ in got) == sorted(
        f"v{i}" for i in range(N) if i % 100 == 7
    )
    assert db.search(vectors[0].tolist(), 5, where={"bucket": 1000}) == []


def test_batch_applies_the_filter_to_every_query(filled):
    db, vectors, _ = filled
    queries = vectors[:3].tolist()
    where = {"bucket": {"$in": [1, 2]}}
    batch = db.search_batch(queries, 5, where=where)
    assert batch == [db.search(q, 5, where=where) for q in queries]


def test_no_filter_is_plain_search(filled):
    db, vectors, _ = filled
    q = vectors[5].tolist()
    assert db.search(q, 5, where=None) == db.search(q, 5)


@pytest.mark.parametrize(
    ("where", "message"),
    [
        ({}, "at least one condition"),
        ({"a": {"$regex": "x"}}, "unknown operator '\\$regex'"),
        ({"$xor": [{"a": 1}]}, "unknown filter operator '\\$xor'"),
        ({"a": [1, 2]}, "\\$in"),
        ({"a": {"$gt": True}}, "number or a string"),
        ({"$and": []}, "non-empty list"),
        ({"a": math.nan}, "where must hold only"),
        ({"a": object()}, "where must hold only"),
        ([("a", 1)], "where must be a dict"),
    ],
)
def test_bad_filters_raise_value_error(filled, where, message):
    db, vectors, _ = filled
    with pytest.raises(ValueError, match=message):
        db.search(vectors[0].tolist(), 3, where=where)
    with pytest.raises(ValueError, match=message):
        db.search_batch([vectors[0].tolist()], 3, where=where)


def test_where_is_keyword_only(filled):
    db, vectors, _ = filled
    with pytest.raises(TypeError):
        db.search(vectors[0].tolist(), 3, {"bucket": 1})
