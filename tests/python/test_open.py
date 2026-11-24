"""srvdb.open() is equivalent to the Database constructor."""

import pytest

import srvdb

DIM = 128


def test_open_returns_database(tmp_path):
    db = srvdb.open(str(tmp_path), dimension=DIM)
    assert isinstance(db, srvdb.Database)
    assert "open" in srvdb.__all__


@pytest.mark.parametrize("mode", ["flat", "hnsw", "ivf", "auto"])
def test_open_accepts_same_arguments(tmp_path, mode):
    db = srvdb.open(str(tmp_path), dimension=DIM, mode=mode)
    db.add(["a"], [[0.1] * DIM], ["{}"])
    db.persist()
    assert db.count() == 1
    positional = srvdb.open(str(tmp_path / "p"), DIM, mode)
    assert isinstance(positional, srvdb.Database)


def test_open_reopens_persisted_data(tmp_path):
    db = srvdb.open(str(tmp_path), dimension=DIM)
    db.add(["a"], [[0.1] * DIM], ['{"k": 1}'])
    db.persist()
    del db
    assert srvdb.open(str(tmp_path), dimension=DIM).get("a") is not None


@pytest.mark.parametrize(
    "kwargs",
    [{"dimension": DIM, "mode": "nope"}, {"dimension": 0}, {"dimension": 100000}],
)
def test_open_raises_same_errors_as_database(tmp_path, kwargs):
    with pytest.raises(ValueError) as via_class:
        srvdb.Database(str(tmp_path / "c"), **kwargs)
    with pytest.raises(ValueError) as via_open:
        srvdb.open(str(tmp_path / "o"), **kwargs)
    assert type(via_open.value) is type(via_class.value)
    assert str(via_open.value).replace("/o", "/c") == str(via_class.value)


def test_recorded_mode_is_used_and_conflicts_raise(tmp_path):
    db = srvdb.open(str(tmp_path), dimension=DIM, mode="sq8")
    vectors = [[float((i * 7 + j) % 13) for j in range(DIM)] for i in range(40)]
    db.add([f"d{i}" for i in range(40)], vectors, ["{}"] * 40)
    db.persist()
    del db
    # The default mode reopens the recorded SQ8 storage, not empty flat storage
    assert srvdb.open(str(tmp_path), dimension=DIM).count() == 40
    with pytest.raises(ValueError, match="settings mismatch"):
        srvdb.open(str(tmp_path), dimension=DIM, mode="hnsw")
    assert srvdb.open(str(tmp_path), dimension=DIM, mode="sq8").count() == 40
