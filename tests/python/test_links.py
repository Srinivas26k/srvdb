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


def test_drop_links_clears_a_closed_database(tmp_path, db):
    db.link("doc0", "cites", "doc1")
    db.save()
    path = str(tmp_path / "db")
    with pytest.raises(BlockingIOError):
        srvdb.drop_links(path)
    db.close()
    assert srvdb.drop_links(path) == 1
    again = srvdb.open(path)
    assert again.neighbors("doc0") == []
    assert again.count() == 6
    again.close()
    with pytest.raises(FileNotFoundError):
        srvdb.drop_links(str(tmp_path / "missing"))


@pytest.fixture
def worded(tmp_path):
    """Six docs with text, chained doc0 -> doc1 -> ... -> doc5 by "next"."""
    database = srvdb.create(str(tmp_path / "worded"), DIM)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    ids = [f"doc{i}" for i in range(6)]
    metas = [json.dumps({"text": w, "even": i % 2 == 0}) for i, w in enumerate(words)]
    database.add(ids, [vec(i) for i in range(6)], metas)
    database.enable_keyword_search("text")
    for i in range(5):
        database.link(f"doc{i}", "next", f"doc{i + 1}")
    yield database
    database.close()


def test_expand_with_vector_search(worded):
    hits = worded.search(vec(0), 1, expand=("next", 1))
    assert [h[2] for h in hits] == [0, 1]
    assert hits[1][0] == f"doc{int(hits[0][0][3:]) + 1}"


def test_expand_with_keyword_only_search(worded):
    hits = worded.search(None, 1, text="charlie", expand=("next", 2))
    assert [(h[0], h[2]) for h in hits] == [("doc2", 0), ("doc3", 1), ("doc4", 2)]
    assert all(h[1] == hits[0][1] for h in hits)


def test_expand_with_hybrid_search(worded):
    plain = worded.search(vec(2), 2, text="charlie")
    hits = worded.search(vec(2), 2, text="charlie", expand=("next", 1))
    assert [(h[0], h[1]) for h in hits[:2]] == plain
    assert all(h[2] == 0 for h in hits[:2])
    assert all(h[2] == 1 for h in hits[2:])


def test_expand_with_where_is_refused(worded):
    with pytest.raises(ValueError, match="where.*expand"):
        worded.search(vec(0), 1, where={"even": True}, expand=("next", 1))


def test_expand_with_text_and_where_is_refused(worded):
    with pytest.raises(ValueError, match="where.*expand"):
        worded.search(None, 1, text="alpha", where={"even": True}, expand=("next", 1))
    with pytest.raises(ValueError, match="where.*expand"):
        worded.search(vec(0), 1, text="alpha", where={"even": True}, expand=("next", 1))


@pytest.fixture
def rabitq_linked(tmp_path):
    database = srvdb.Database.new_rabitq(str(tmp_path / "rq"), dimension=DIM, bits=4)
    database.add([f"doc{i}" for i in range(6)], [vec(i) for i in range(6)], ["{}"] * 6)
    database.link("doc0", "next", "doc1")
    yield database
    database.close()


def test_expand_with_certify_is_refused(rabitq_linked):
    assert rabitq_linked.search(vec(0), 2, certify=True).certified >= 0
    with pytest.raises(ValueError, match="expand.*certify"):
        rabitq_linked.search(vec(0), 2, certify=True, expand=("next", 1))


def test_expand_with_deadline_is_refused(rabitq_linked):
    assert len(rabitq_linked.search(vec(0), 2, deadline_ms=60_000)) == 2
    with pytest.raises(ValueError, match="expand.*deadline_ms"):
        rabitq_linked.search(vec(0), 2, deadline_ms=60_000, expand=("next", 1))


def test_expand_works_in_rabitq_mode_without_certify(rabitq_linked):
    hits = rabitq_linked.search(vec(0), 1, expand=("next", 1))
    assert [h[2] for h in hits][:1] == [0]


def test_expand_without_query_or_text_is_refused(worded):
    with pytest.raises(ValueError, match="query vector, text"):
        worded.search(None, 1, expand=("next", 1))
