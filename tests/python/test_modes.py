"""End-to-end tests of the Python package against the built extension."""

import json
import random

import pytest

import srvdb

DIM = 128
N = 300


def random_vectors(n, seed=0):
    rng = random.Random(seed)
    return [[rng.random() - 0.5 for _ in range(DIM)] for _ in range(n)]


@pytest.fixture
def vectors():
    return random_vectors(N)


def fill(db, vectors):
    ids = [f"doc{i}" for i in range(len(vectors))]
    metas = [json.dumps({"i": i}) for i in range(len(vectors))]
    db.add(ids, vectors, metas)
    return ids


@pytest.mark.parametrize("mode", ["flat", "hnsw", "sq8", "pq"])
def test_mode_finds_vectors_and_survives_reopen(tmp_path, vectors, mode):
    db = srvdb.Database(str(tmp_path), dimension=DIM, mode=mode)
    fill(db, vectors)
    db.persist()
    del db

    db = srvdb.Database(str(tmp_path), dimension=DIM, mode=mode)
    assert db.count() == N
    hits = sum(db.search(vectors[i], 1)[0][0] == f"doc{i}" for i in range(20))
    assert hits >= 19


def test_metadata_round_trip(tmp_path, vectors):
    db = srvdb.Database(str(tmp_path), dimension=DIM)
    fill(db, vectors)
    meta = json.loads(db.get("doc7"))
    assert meta["i"] == 7
    assert meta["__id__"] == "doc7"


def test_delete_persists(tmp_path, vectors):
    db = srvdb.Database(str(tmp_path), dimension=DIM)
    fill(db, vectors)
    assert db.delete(["doc3", "missing"]) == 1
    db.persist()
    del db

    db = srvdb.Database(str(tmp_path), dimension=DIM)
    assert "doc3" not in db.get_all_ids()
    assert all(doc != "doc3" for doc, _ in db.search(vectors[3], 5))


def test_search_sees_unpersisted_vectors(tmp_path, vectors):
    db = srvdb.Database(str(tmp_path), dimension=DIM)
    fill(db, vectors)
    assert db.search(vectors[42], 1)[0][0] == "doc42"


def test_rejects_wrong_dimension(tmp_path):
    db = srvdb.Database(str(tmp_path), dimension=DIM)
    with pytest.raises(ValueError):
        db.add(["a"], [[0.0] * (DIM + 1)], ["{}"])
    with pytest.raises(ValueError):
        db.search([0.0] * (DIM - 1), 1)


def test_rejects_duplicate_ids(tmp_path, vectors):
    db = srvdb.Database(str(tmp_path), dimension=DIM)
    fill(db, vectors[:5])
    with pytest.raises(ValueError):
        db.add(["doc1"], [vectors[0]], ["{}"])


def test_pq_mode_requires_large_first_batch(tmp_path, vectors):
    db = srvdb.Database(str(tmp_path), dimension=DIM, mode="pq")
    with pytest.raises(ValueError, match="at least 256"):
        fill(db, vectors[:10])


def test_old_class_name_still_works_with_warning(tmp_path):
    with pytest.warns(DeprecationWarning, match="srvdb.Database"):
        cls = srvdb.SrvDBPython
    assert cls is srvdb.Database
    assert cls(str(tmp_path), dimension=DIM).count() == 0


def test_set_ef_search(tmp_path):
    db = srvdb.Database.new_hnsw(str(tmp_path), dimension=DIM)
    db.set_ef_search(200)
    with pytest.raises(ValueError, match="at least 1"):
        db.set_ef_search(0)
