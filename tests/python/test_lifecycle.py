"""db.close(), the context manager and srvdb.delete (ADR 0013)."""

import pytest

import srvdb

DIM = 128


def fill(db, n=5, start=0):
    ids = [f"d{i}" for i in range(start, start + n)]
    db.add(ids, [[float(i % 7) + 1.0] * DIM for i in range(n)], ["{}"] * n)


def test_close_releases_the_lock_and_is_idempotent(tmp_path):
    path = str(tmp_path / "db")
    db = srvdb.create(path, DIM)
    db.close()
    db.close()
    assert repr(db) == "SrvDB(closed)"
    with pytest.raises(ValueError, match="closed"):
        db.count()
    srvdb.open(path).close()


def test_close_does_not_save(tmp_path):
    path = str(tmp_path / "db")
    db = srvdb.create(path, DIM)
    fill(db)
    db.close()
    assert srvdb.open(path).count() == 0


def test_with_block_saves_and_closes(tmp_path):
    path = str(tmp_path / "db")
    with srvdb.create(path, DIM) as db:
        fill(db)
    with pytest.raises(ValueError, match="closed"):
        db.count()
    with srvdb.open(path) as db:
        assert db.count() == 5
        fill(db, start=5)
    with srvdb.open(path) as db:
        assert db.count() == 10


def test_with_block_on_exception_closes_without_saving(tmp_path):
    path = str(tmp_path / "db")
    with srvdb.create(path, DIM) as db:
        fill(db)
    with pytest.raises(KeyError):
        with srvdb.open(path) as db:
            fill(db, start=5)
            raise KeyError("half-finished batch")
    # The lock is released and the unsaved batch is not durable
    with srvdb.open(path) as db:
        assert db.count() == 5


def test_delete_removes_a_closed_database(tmp_path):
    path = tmp_path / "db"
    with srvdb.create(str(path), DIM) as db:
        fill(db)
    srvdb.delete(str(path))
    assert not path.exists()
    with pytest.raises(FileNotFoundError):
        srvdb.delete(str(path))


def test_delete_refuses_a_locked_database(tmp_path):
    path = tmp_path / "db"
    db = srvdb.create(str(path), DIM)
    with pytest.raises(BlockingIOError, match="already open"):
        srvdb.delete(str(path))
    assert path.exists()
    db.close()
    srvdb.delete(str(path))


def test_delete_refuses_a_directory_with_other_files(tmp_path):
    path = tmp_path / "db"
    srvdb.create(str(path), DIM).close()
    (path / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="notes.txt"):
        srvdb.delete(str(path))
    assert (path / "notes.txt").read_text() == "keep me"


def test_delete_refuses_a_non_srvdb_directory(tmp_path):
    (tmp_path / "metadata.db").write_text("someone else's file")
    with pytest.raises(ValueError, match="not a SrvDB database"):
        srvdb.delete(str(tmp_path))
    assert (tmp_path / "metadata.db").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="not a SrvDB database"):
        srvdb.delete(str(empty))
