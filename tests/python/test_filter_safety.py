"""Hardening of where= filters: caps, key types, exact integers, messages."""

import json

import pytest

import srvdb

DIM = 128


@pytest.fixture
def db(tmp_path):
    db = srvdb.create(str(tmp_path), DIM, "flat")
    big = 2**53 + 1
    vectors = [[float(i == j) for j in range(DIM)] for i in range(3)]
    metas = [{"n": big}, {"n": big - 1}, {"n": 1, "a.b": "dotted"}]
    db.add(["big", "below", "one"], vectors, [json.dumps(m) for m in metas])
    yield db
    db.close()


def q():
    return [1.0] * DIM


def ids(hits):
    return sorted(id_ for id_, _ in hits)


def test_integers_above_2_53_compare_exactly(db):
    big = 2**53 + 1
    assert ids(db.search(q(), 5, where={"n": big})) == ["big"]
    assert ids(db.search(q(), 5, where={"n": {"$in": [big]}})) == ["big"]
    assert ids(db.search(q(), 5, where={"n": {"$gt": big - 1}})) == ["big"]
    assert ids(db.search(q(), 5, where={"n": float(big - 1)})) == ["below"]


class Secret:
    def __repr__(self):
        return "secret-repr-value"


@pytest.mark.parametrize("value", [Secret(), float("nan"), {"x": Secret()}])
def test_encoding_errors_do_not_echo_values(db, value):
    with pytest.raises(ValueError) as err:
        db.search(q(), 3, where={"a": value})
    message = str(err.value)
    assert message.startswith("where must hold only strings, numbers")
    assert "secret" not in message
    assert "nan" not in message.lower().replace("no nan", "")


@pytest.mark.parametrize(
    "where",
    [
        {1: "x"},
        {"a": {1: 2}},
        {"$or": [{"a": 1}, {None: 2}]},
        {"$and": ({"a": 1}, {True: 1})},
    ],
)
def test_non_str_keys_are_rejected_at_any_level(db, where):
    with pytest.raises(ValueError, match="dict key that is not a str"):
        db.search(q(), 3, where=where)


def test_deep_or_self_referencing_where_is_rejected(db):
    loop = {"a": {}}
    loop["a"]["$not"] = loop
    with pytest.raises(ValueError, match="deeper than 64"):
        db.search(q(), 3, where=loop)


def test_list_and_condition_caps(db):
    ok = {"n": {"$in": list(range(10_000))}}
    assert ids(db.search(q(), 5, where=ok)) == ["one"]
    with pytest.raises(ValueError, match="10001 values; at most 10000"):
        db.search(q(), 5, where={"n": {"$nin": list(range(10_001))}})
    many = {"$or": [{"n": i} for i in range(1_001)]}
    with pytest.raises(ValueError, match="at most 1000 conditions"):
        db.search(q(), 5, where=many)
