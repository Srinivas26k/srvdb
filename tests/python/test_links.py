"""Graph links and search expansion (ADR 0016)."""

import json

import pytest

import srvdb

DIM = 128


def vec(i: int) -> list[float]:
    return [float((i * 7 + j * 3) % 11) - 5.0 + i * 0.01 for j in range(DIM)]


@pytest.fixture
def db(tmp_path):
    database = srvdb.create(str(tmp_path / "db"), DIM)
    ids = [f"doc{i}" for i in range(6)]
    database.add(ids, [vec(i) for i in range(6)], [json.dumps({"i": i}) for i in range(6)])
    database.save()
    yield database
    database.close()


def test_link_neighbors_and_reopen(tmp_path, db):
    assert db.link("doc0", "cites", "doc1") is True
    assert db.link("doc0", "cites", "doc1") is False
    db.link("doc0", "about", "doc2")
    assert db.neighbors("doc0") == [("about", "doc2"), ("cites", "doc1")]
    assert db.neighbors("doc0", type="cites") == [("cites", "doc1")]
    db.save()
    db.link("doc3", "cites", "doc4")  # not saved
    db.close()
    again = srvdb.open(str(tmp_path / "db"))
    assert again.neighbors("doc0") == [("about", "doc2"), ("cites", "doc1")]
    assert again.neighbors("doc3") == []
    assert again.unlink("doc0", "about", "doc2") is True
    assert again.unlink("doc0", "about", "doc2") is False
    assert again.unlink("missing", "about", "doc2") is False
    again.close()


def test_delete_cascades_and_upsert_keeps_links(db):
    db.link("doc0", "cites", "doc1")
    db.link("doc1", "cites", "doc2")
    db.upsert(["doc1"], [vec(1)], ["{}"])
    assert db.neighbors("doc0") == [("cites", "doc1")]
    assert db.neighbors("doc1") == [("cites", "doc2")]
    db.delete(["doc1"])
    assert db.neighbors("doc0") == []
    db.compact()
    assert db.neighbors("doc0") == []


def test_invalid_links_raise_value_error(db):
    for args in [
        ("doc0", "Cites", "doc1"),
        ("doc0", "cites", "nope"),
        ("doc0", "cites", "doc0"),
    ]:
        with pytest.raises(ValueError):
            db.link(*args)
    with pytest.raises(ValueError):
        db.neighbors("nope")


def test_search_expand_marks_hops(db):
    for i in range(5):
        db.link(f"doc{i}", "next", f"doc{i + 1}")
    plain = db.search(vec(0), 1)
    hits = db.search(vec(0), 1, expand=("next", 2))
    assert hits[0] == (plain[0][0], plain[0][1], 0)
    assert [h[2] for h in hits] == list(range(len(hits)))
    assert len(hits) <= 3
    assert len(db.search(vec(0), 1, expand=("next", 3, 1))) == 2
    for bad in [("next", 0), ("next", 4), ("next",), "next", ("next", 1, 0)]:
        with pytest.raises(ValueError):
            db.search(vec(0), 1, expand=bad)
    with pytest.raises(ValueError):
        db.search(vec(0), 1, where={"i": 0}, expand=("next", 1))
