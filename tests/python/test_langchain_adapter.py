"""LangChain adapter (srvdb.integrations.langchain); skipped without the extra."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

import srvdb  # noqa: E402
from fake_embedding import DIM, embed  # noqa: E402
from srvdb.integrations.langchain import SrvDBVectorStore  # noqa: E402

TEXTS = [
    "the cat sat on the mat",
    "dogs chase the ball in the park",
    "rust is a systems programming language",
    "python is popular for machine learning",
]
METAS = [
    {"topic": "pets"},
    {"topic": "pets"},
    {"topic": "code"},
    {"topic": "code"},
]


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [embed(t) for t in texts]

    def embed_query(self, text):
        return embed(text)


@pytest.fixture
def store(tmp_path):
    return SrvDBVectorStore.from_texts(
        TEXTS, FakeEmbeddings(), METAS, ids=["a", "b", "c", "d"], path=str(tmp_path)
    )


def test_search_returns_best_match_with_cosine_score(store):
    hits = store.similarity_search_with_score("rust systems language", k=2)
    assert hits[0][0].page_content == TEXTS[2]
    assert hits[0][0].metadata == {"topic": "code"}
    assert hits[0][0].id == "c"
    assert hits[0][1] >= hits[1][1]
    assert -1.0 <= hits[1][1] <= 1.0 + 1e-6


def test_dict_and_callable_filters(store):
    docs = store.similarity_search(
        "rust systems language", k=2, filter={"topic": "pets"}
    )
    assert {d.id for d in docs} == {"a", "b"}
    docs = store.similarity_search("cat", k=4, filter=lambda m: m["topic"] == "code")
    assert {d.id for d in docs} == {"c", "d"}
    assert store.similarity_search("cat", k=3, filter={"topic": "none"}) == []


def test_delete_and_get_by_ids(store):
    assert store.delete(["a", "missing"])
    assert [d.id for d in store.get_by_ids(["a", "b"])] == ["b"]
    assert "a" not in {d.id for d in store.similarity_search("cat mat", k=4)}


def test_add_documents_keeps_ids_and_reopens(tmp_path):
    store = SrvDBVectorStore(srvdb.open(str(tmp_path), dimension=DIM), FakeEmbeddings())
    ids = store.add_documents(
        [Document(id="x", page_content="hello world", metadata={"n": 1})]
    )
    assert ids == ["x"]
    del store
    reopened = SrvDBVectorStore(
        srvdb.open(str(tmp_path), dimension=DIM), FakeEmbeddings()
    )
    assert reopened.get_by_ids(["x"])[0].metadata == {"n": 1}


def test_retriever_and_relevance_scores(store):
    docs = store.as_retriever(search_kwargs={"k": 1}).invoke("dogs ball park")
    assert docs[0].id == "b"
    scored = store.similarity_search_with_relevance_scores("dogs ball park", k=4)
    assert all(0.0 <= s <= 1.0 for _, s in scored)
