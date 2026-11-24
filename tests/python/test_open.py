"""srvdb.create, srvdb.open, db.save and srvdb.delete (ADR 0013)."""

import pytest

import srvdb

DIM = 128


def rows(n, seed=0):
    return [
        [float((i * 7 + j * 3 + seed) % 13) + 1.0 for j in range(DIM)] for i in range(n)
    ]


def fill(db, n=20):
    db.add([f"d{i}" for i in range(n)], rows(n), ["{}"] * n)


def test_verbs_are_exported():
    for name in ("create", "open", "delete"):
        assert name in srvdb.__all__


@pytest.mark.parametrize(
    ("index", "options"),
    [
        ("flat", {}),
        ("hnsw", {"m": 8, "ef_construction": 64, "ef_search": 32}),
        ("ivf", {"nlist": 4, "nprobe": 2}),
        ("sq8", {}),
        ("sq8", {"training": rows(50)}),
        ("pq", {"training": rows(300)}),
        ("rabitq", {"bits": 2, "rerank": 5}),
        ("auto", {}),
    ],
)
def test_create_then_open_by_path_alone(tmp_path, index, options):
    path = str(tmp_path / "db")
    db = srvdb.create(path, DIM, index, **options)
    fill(db)
    db.save()
    db.close()
    reopened = srvdb.open(path)
    assert reopened.count() == 20
    assert f"mode='{index}'" in reopened.info()
    assert reopened.search(rows(1)[0], k=3)


def test_default_index_is_hnsw(tmp_path):
    db = srvdb.create(str(tmp_path / "db"), dimension=DIM)
    assert "mode='hnsw'" in repr(db)


def test_persist_is_an_alias_of_save(tmp_path):
    path = str(tmp_path / "db")
    db = srvdb.create(path, DIM)
    fill(db, 3)
    db.persist()
    db.close()
    assert srvdb.open(path).count() == 3


def test_create_refuses_existing_database(tmp_path):
    path = str(tmp_path / "db")
    srvdb.create(path, DIM).close()
    with pytest.raises(FileExistsError, match="already exists"):
        srvdb.create(path, DIM)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "notes.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not empty"):
        srvdb.create(str(tmp_path / "other"), DIM)


def test_create_exist_ok_opens_only_matching_settings(tmp_path):
    path = str(tmp_path / "db")
    db = srvdb.create(path, DIM, "hnsw", m=8)
    fill(db, 5)
    db.save()
    db.close()
    with srvdb.create(path, DIM, "hnsw", exist_ok=True, m=8) as again:
        assert again.count() == 5
    with pytest.raises(ValueError, match="m = 8, not 12"):
        srvdb.create(path, DIM, "hnsw", exist_ok=True, m=12)
    with pytest.raises(ValueError, match="'hnsw', not 'flat'"):
        srvdb.create(path, DIM, "flat", exist_ok=True)
    with pytest.raises(ValueError, match="dimension mismatch"):
        srvdb.create(path, DIM * 2, "hnsw", exist_ok=True)
    # exist_ok on a new path simply creates
    srvdb.create(str(tmp_path / "new"), DIM, exist_ok=True).close()


@pytest.mark.parametrize(
    ("index", "options", "message"),
    [
        ("hnsw", {"bits": 4}, "valid options: m, ef_construction, ef_search"),
        ("flat", {"m": 16}, "takes no options"),
        ("rabitq", {"bits": 3}, "bits"),
        ("hnsw", {"m": 1}, "m must be between 2 and 256"),
        ("hnsw", {"m": "big"}, "must be a non-negative integer"),
        ("nope", {}, "choose one of: flat, hnsw, ivf, sq8, pq, rabitq, auto"),
        ("sq8", {"training": "x"}, "training must be a sequence"),
    ],
)
def test_create_validates_index_and_options(tmp_path, index, options, message):
    path = tmp_path / "db"
    with pytest.raises(ValueError, match=message):
        srvdb.create(str(path), DIM, index, **options)
    assert not path.exists(), "a refused create writes nothing"


def test_create_rejects_bad_dimension(tmp_path):
    with pytest.raises(ValueError, match="dimension must be between 1 and 4096"):
        srvdb.create(str(tmp_path / "db"), 0)


def test_open_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="create the database first"):
        srvdb.open(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        srvdb.open(str(tmp_path))
    assert not (tmp_path / "missing").exists()


def test_open_locked_database(tmp_path):
    path = str(tmp_path / "db")
    db = srvdb.create(path, DIM)
    with pytest.raises(BlockingIOError):
        srvdb.open(path)
    db.close()
    srvdb.open(path).close()


def test_database_class_still_works(tmp_path):
    db = srvdb.Database(str(tmp_path / "db"), dimension=DIM, mode="hnsw")
    assert isinstance(db, srvdb.Database)
    assert isinstance(srvdb.open, type(srvdb.create))
