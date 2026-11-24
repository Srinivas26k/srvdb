"""Keyword (BM25) and hybrid search from Python (#79, ADR 0015)."""

import json
import math
import re

import pytest

import srvdb

DIM = 8
TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a quick fox",
    "Lazy afternoons: the dog sleeps",
    "Rust error E0308: mismatched types",
    "Foxes and dogs, a field guide",
]


def vec(i):
    return [float((i * 3 + j) % 7 + 1) for j in range(DIM)]


def tokens(text):
    return [t.lower() for t in re.split(r"\W|_", text) if t]


def bm25(docs, query):
    """Textbook BM25 (k1=1.2, b=0.75, Lucene idf); docs: {id: text}."""
    toks = {i: tokens(t) for i, t in docs.items()}
    n = len(toks)
    avgdl = sum(len(t) for t in toks.values()) / n
    scores = {}
    for i, words in toks.items():
        s = 0.0
        for term in set(tokens(query)):
            tf = words.count(term)
            if not tf:
                continue
            df = sum(term in w for w in toks.values())
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(words) / avgdl))
        if s > 0:
            scores[i] = s
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@pytest.fixture
def db(tmp_path):
    db = srvdb.create(str(tmp_path / "kw"), DIM)
    ids = [f"d{i}" for i in range(len(TEXTS))]
    metas = [json.dumps({"text": t, "even": i % 2 == 0}) for i, t in enumerate(TEXTS)]
    db.add(ids, [vec(i) for i in range(len(TEXTS))], metas)
    db.enable_keyword_search("text")
    yield db
    db.close()


def test_keyword_only_matches_reference(db):
    docs = {f"d{i}": t for i, t in enumerate(TEXTS)}
    got = db.search(None, 10, text="quick fox")
    want = bm25(docs, "quick fox")
    assert [g[0] for g in got] == [w[0] for w in want]
    for (_, score), (_, ref) in zip(got, want):
        assert score == pytest.approx(ref, rel=1e-4)


def test_hybrid_fuses_both_rankings(db):
    got = db.search(vec(3), 3, text="E0308")
    ids = [g[0] for g in got]
    assert ids[0] == "d3"  # first in both lists
    assert got[0][1] == pytest.approx(2 / 61, rel=1e-4)


def test_where_applies_to_text(db):
    got = db.search(None, 10, text="dog", where={"even": True})
    assert got and all(int(i[1:]) % 2 == 0 for i, _ in got)


def test_upsert_delete_and_reopen(tmp_path, db):
    db.delete(["d1"])
    db.upsert(["d4"], [vec(4)], [json.dumps({"text": "zebra crossing"})])
    assert [g[0] for g in db.search(None, 5, text="zebra")] == ["d4"]
    assert "d1" not in [g[0] for g in db.search(None, 5, text="outpaces")]
    db.save()
    db.close()
    again = srvdb.open(str(tmp_path / "kw"))
    try:
        assert [g[0] for g in again.search(None, 5, text="zebra")] == ["d4"]
        assert again.search(None, 5, text="foxes") == []
    finally:
        again.close()


def test_errors(tmp_path, db):
    with pytest.raises(ValueError):
        db.search(None, 3)
    with pytest.raises(ValueError):
        db.search(None, 3, text="  ,, ")
    with pytest.raises(ValueError):
        db.search(None, 0, text="fox")
    with pytest.raises(ValueError):
        db.enable_keyword_search("")
    plain = srvdb.create(str(tmp_path / "plain"), DIM)
    try:
        with pytest.raises(ValueError, match="enable_keyword"):
            plain.search(vec(0), 3, text="fox")
    finally:
        plain.close()
