"""Input validation and exception types raised by the Python API."""

import math

import pytest

import srvdb

DIM = 128


@pytest.fixture
def db(tmp_path):
    return srvdb.Database(str(tmp_path), dimension=DIM)


def vec(value=0.1):
    return [value] * DIM


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(db, bad):
    v = vec()
    v[3] = bad
    with pytest.raises(ValueError, match="NaN or infinity"):
        db.add(["a"], [v], ["{}"])
    with pytest.raises(ValueError, match="NaN or infinity"):
        db.search(v, 1)
    assert db.count() == 0


def test_k_must_be_positive(db):
    db.add(["a"], [vec()], ["{}"])
    with pytest.raises(ValueError, match="k must be at least 1"):
        db.search(vec(), 0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        db.search_batch([vec()], 0)


@pytest.mark.parametrize(
    ("meta", "message"),
    [("not json", "not valid JSON"), ("[1, 2]", "must be a JSON object")],
)
def test_metadata_must_be_a_json_object(db, meta, message):
    with pytest.raises(ValueError, match=message):
        db.add(["a"], [vec()], [meta])
    assert db.count() == 0


def test_ids_must_be_non_empty_and_unique_in_batch(db):
    with pytest.raises(ValueError, match="empty"):
        db.add([""], [vec()], ["{}"])
    with pytest.raises(ValueError, match="duplicate id 'a' within this batch"):
        db.add(["a", "a"], [vec(), vec(0.2)], ["{}", "{}"])
    assert db.count() == 0


def test_mismatched_lengths_are_rejected(db):
    with pytest.raises(ValueError, match="same length"):
        db.add(["a", "b"], [vec()], ["{}", "{}"])


def test_failed_batch_stores_nothing(db):
    bad = vec()
    bad[0] = math.nan
    with pytest.raises(ValueError):
        db.add(["a", "b"], [vec(), bad], ["{}", "{}"])
    assert db.count() == 0
    assert db.get_all_ids() == []
