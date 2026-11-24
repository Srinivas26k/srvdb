"""The framework adapters pass filters to SrvDB instead of post-filtering (#46)."""

import pytest

import srvdb
from fake_embedding import DIM, embed

N = 300


class Spy:
    """Wraps a Database and records the ``where`` and ``k`` of each search."""

    def __init__(self, db):
        self._db = db
        self.searches = []

    def search(self, query, k, *, where=None):
        self.searches.append((k, where))
        return self._db.search(query, k, where=where)

    def __getattr__(self, name):
        return getattr(self._db, name)


@pytest.fixture
def spy(tmp_path):
    db = srvdb.create(str(tmp_path), DIM, "hnsw")
    yield Spy(db)
    db.close()


def langchain_store(spy):
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import Embeddings

    from srvdb.integrations.langchain import SrvDBVectorStore

    class Fake(Embeddings):
        def embed_documents(self, texts):
            return [embed(t) for t in texts]

        def embed_query(self, text):
            return embed(text)

    store = SrvDBVectorStore(spy, Fake())
    texts = [f"note {i} about topic {i % 7}" for i in range(N)]
    metas = [{"n": i, "kind": "rare" if i % 100 == 0 else "common"} for i in range(N)]
    store.add_texts(texts, metas, ids=[f"d{i}" for i in range(N)])
    return store


def test_langchain_dict_filter_runs_in_srvdb(spy):
    store = langchain_store(spy)
    docs = store.similarity_search("note topic", k=3, filter={"kind": "rare"})
    assert sorted(d.metadata["n"] for d in docs) == [0, 100, 200]
    assert spy.searches[-1] == (3, {"metadata.kind": "rare"})


def test_langchain_operator_filters(spy):
    store = langchain_store(spy)
    docs = store.similarity_search(
        "note", k=50, filter={"$or": [{"n": {"$lt": 2}}, {"n": {"$gte": 298}}]}
    )
    assert sorted(d.metadata["n"] for d in docs) == [0, 1, 298, 299]
    assert spy.searches[-1][1] == {
        "$or": [{"metadata.n": {"$lt": 2}}, {"metadata.n": {"$gte": 298}}]
    }
    with pytest.raises(ValueError, match="unknown operator"):
        store.similarity_search("note", k=2, filter={"n": {"$near": 1}})


def test_langchain_callable_filter_still_post_filters(spy):
    store = langchain_store(spy)
    docs = store.similarity_search("note", k=2, filter=lambda m: m["n"] in (5, 6))
    assert sorted(d.metadata["n"] for d in docs) == [5, 6]
    assert all(where is None for _, where in spy.searches)


def test_llamaindex_filters_node_ids_and_doc_ids_run_in_srvdb(spy):
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
    from llama_index.core.vector_stores.types import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
    )

    from srvdb.integrations.llamaindex import SrvDBVectorStore

    store = SrvDBVectorStore(spy)
    nodes = []
    for i in range(N):
        node = TextNode(id_=f"n{i}", text=f"chunk {i}", metadata={"n": i})
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=f"doc{i % 3}"
        )
        node.embedding = embed(f"chunk {i % 11}")
        nodes.append(node)
    store.add(nodes)

    filters = MetadataFilters(
        filters=[MetadataFilter(key="n", value=10, operator=FilterOperator.LT)]
    )
    q = VectorStoreQuery(
        query_embedding=embed("chunk 1"),
        similarity_top_k=20,
        filters=filters,
        doc_ids=["doc0"],
    )
    result = store.query(q)
    assert sorted(result.ids) == ["n0", "n3", "n6", "n9"]
    where = spy.searches[-1][1]
    assert {"metadata.n": {"$lt": 10}} in where["$and"][-1]["$and"]
    assert {"metadata.ref_doc_id": {"$in": ["doc0"]}} in where["$and"]

    q = VectorStoreQuery(
        query_embedding=embed("chunk 1"), similarity_top_k=5, node_ids=["n7", "n8"]
    )
    assert sorted(store.query(q).ids) == ["n7", "n8"]
    assert spy.searches[-1][1] == {"$and": [{"__id__": {"$in": ["n7", "n8"]}}]}
