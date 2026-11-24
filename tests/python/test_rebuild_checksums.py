"""srvdb.rebuild_checksums: explicit recovery of missing .crc files (#33)."""

import pytest

import srvdb

DIM = 128


def _db(path):
    with srvdb.create(str(path), dimension=DIM, index="flat") as db:
        db.add(["a", "b"], [[1.0] + [0.0] * 127, [0.0, 1.0] + [0.0] * 126], ["{}", "{}"])


def test_missing_sidecar_is_rebuilt(tmp_path):
    _db(tmp_path / "db")
    (tmp_path / "db" / "vectors.bin.crc").unlink()
    with pytest.raises(RuntimeError):
        srvdb.open(str(tmp_path / "db"))
    assert srvdb.rebuild_checksums(str(tmp_path / "db")) == ["vectors.bin.crc"]
    with srvdb.open(str(tmp_path / "db")) as db:
        assert db.count() == 2
    assert srvdb.rebuild_checksums(str(tmp_path / "db")) == []


def test_damaged_data_is_refused(tmp_path):
    _db(tmp_path / "db")
    path = tmp_path / "db" / "vectors.bin"
    data = bytearray(path.read_bytes())
    data[30] ^= 0x04
    path.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="backup"):
        srvdb.rebuild_checksums(str(tmp_path / "db"))


def test_open_database_is_refused(tmp_path):
    _db(tmp_path / "db")
    db = srvdb.open(str(tmp_path / "db"))
    with pytest.raises(BlockingIOError):
        srvdb.rebuild_checksums(str(tmp_path / "db"))
    db.close()


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        srvdb.rebuild_checksums(str(tmp_path / "nope"))
