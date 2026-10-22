"""Delete and upsert through the Python API (ADR 0004)."""

import json
import random

import pytest

import srvdb

DIM = 128


def vectors(n, seed=1):
    rng = random.Random(seed)
    return [[rng.random() - 0.5 for _ in range(DIM)] for _ in range(n)]


@pytest.fixture
def db(tmp_path):
    database = srvdb.Database(str(tmp_path), dimension=DIM)
    vs = vectors(200)
    database.add([f"doc{i}" for i in range(200)], vs, ["{}"] * 200)
    return database, vs


def test_deleted_vectors_never_return_and_top_k_stays_full(db):
    database, vs = db
    nearest = [doc for doc, _ in database.search(vs[5], 10)]
    assert database.delete(nearest) == 10
    results = database.search(vs[5], 10)
    assert len(results) == 10
    assert not set(nearest) & {doc for doc, _ in results}
    assert database.count() == 190


def test_upsert_replaces_existing_ids_and_adds_new_ones(db):
    database, vs = db
    fresh = vectors(2, seed=9)
    written = database.upsert(
        ["doc3", "new"], fresh, [json.dumps({"v": 2}), json.dumps({"v": 1})]
    )
    assert written == 2
    assert database.count() == 201
    assert json.loads(database.get("doc3"))["v"] == 2
    assert database.search(fresh[0], 1)[0][0] == "doc3"
    assert all(doc != "doc3" for doc, _ in database.search(vs[3], 5))


def test_add_refuses_existing_ids(db):
    database, vs = db
    with pytest.raises(ValueError, match="upsert"):
        database.add(["doc1"], [vs[0]], ["{}"])


def test_invalid_upsert_changes_nothing(db):
    database, vs = db
    with pytest.raises(ValueError):
        database.upsert(["doc1", "doc2"], [vs[0], vs[1]], ["{}", "not json"])
    assert database.count() == 200
    assert database.search(vs[1], 1)[0][0] == "doc1"


def test_delete_and_upsert_persist(tmp_path):
    database = srvdb.Database(str(tmp_path), dimension=DIM)
    vs = vectors(20)
    database.add([f"d{i}" for i in range(20)], vs, ["{}"] * 20)
    database.delete(["d0"])
    database.upsert(["d1"], [vs[19]], ['{"v": 2}'])
    database.persist()
    del database

    database = srvdb.Database(str(tmp_path), dimension=DIM)
    assert database.count() == 19
    assert "d0" not in database.get_all_ids()
    assert json.loads(database.get("d1"))["v"] == 2
