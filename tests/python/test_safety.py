"""Security review fixes: failed create, symlinks, negative counts, get()."""

import os

import pytest

import srvdb

DIM = 128


def test_failed_create_leaves_nothing_and_retry_succeeds(tmp_path):
    path = tmp_path / "db"
    # PQ needs a dimension divisible by 4; training fails after files exist
    bad = [[float(i % 7) + 1.0] * 130 for i in range(300)]
    with pytest.raises(ValueError, match="divisible by 4"):
        srvdb.create(str(path), 130, "pq", training=bad)
    assert not path.exists()
    assert os.listdir(tmp_path) == []
    with srvdb.create(str(path), DIM, "flat") as db:
        db.add(["a"], [[1.0] * DIM], ["{}"])
    assert srvdb.open(str(path)).count() == 1


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_delete_refuses_a_symlink(tmp_path):
    target = tmp_path / "db"
    srvdb.create(str(target), DIM).close()
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="symbolic link"):
        srvdb.delete(str(link))
    assert (target / "vectors.bin").exists()
    srvdb.delete(str(target))
    assert os.listdir(tmp_path) == ["link"], "no tombstone left"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_fifo_does_not_hang_create(tmp_path):
    os.mkfifo(tmp_path / "index_settings.bin")
    with pytest.raises(FileExistsError, match="not empty"):
        srvdb.create(str(tmp_path), DIM)


@pytest.fixture
def db(tmp_path):
    d = srvdb.create(str(tmp_path / "db"), DIM, "hnsw")
    d.add(["a"], [[1.0] * DIM], ['{"t": 1}'])
    return d


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda d: d.search([1.0] * DIM, k=-1), "k"),
        (lambda d: d.search_batch([[1.0] * DIM], k=-1), "k"),
        (lambda d: d.set_ef_search(-5), "ef_search"),
        (lambda d: d.configure_ivf(nlist=-1), "nlist"),
    ],
)
def test_negative_counts_raise_value_error(db, call, name):
    with pytest.raises(ValueError, match=f"{name} must be a non-negative integer"):
        call(db)


def test_negative_rerank_raises_value_error(tmp_path):
    d = srvdb.create(str(tmp_path / "r"), DIM, "rabitq")
    with pytest.raises(ValueError, match="rerank must be a non-negative integer"):
        d.set_rabitq_rerank(-1)


def test_get_unknown_or_deleted_id_returns_none(db):
    assert db.get("missing") is None
    db.delete(["a"])
    assert db.get("a") is None
