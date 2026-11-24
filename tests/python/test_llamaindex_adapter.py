"""LlamaIndex adapter (srvdb.integrations.llamaindex); skipped without the extra."""

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode  # noqa: E402
from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

import srvdb  # noqa: E402
from fake_embedding import DIM, embed  # noqa: E402
from srvdb.integrations.llamaindex import SrvDBVectorStore  # noqa: E402

ROWS = [
    ("n1", "doc1", "the cat sat on the mat", {"topic": "pets", "year": 2020}),
    ("n2", "doc1", "dogs chase the ball in the park", {"topic": "pets", "year": 2022}),
    (
        "n3",
        "doc2",
        "rust is a systems programming language",
        {"topic": "code", "year": 2024},
    ),
]


def node(node_id, doc_id, text, meta):
    n = TextNode(id_=node_id, text=text, metadata=meta, embedding=embed(text))
    n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return n


@pytest.fixture
def store(tmp_path):
    s = SrvDBVectorStore(srvdb.create(str(tmp_path), DIM))
    assert s.add([node(*r) for r in ROWS]) == ["n1", "n2", "n3"]
    return s


def query(store, text, k=3, **kw):
    return store.query(
        VectorStoreQuery(query_embedding=embed(text), similarity_top_k=k, **kw)
    )


def test_query_returns_nodes_text_metadata_and_scores(store):
    res = query(store, "rust systems language", k=2)
    assert res.ids[0] == "n3"
    assert res.nodes[0].get_content() == ROWS[2][2]
    assert res.nodes[0].metadata == ROWS[2][3]
    assert res.nodes[0].ref_doc_id == "doc2"
    assert res.similarities[0] >= res.similarities[1]


def test_metadata_filters(store):
    f = MetadataFilters(filters=[MetadataFilter(key="topic", value="pets")])
    assert set(query(store, "rust", filters=f).ids) == {"n1", "n2"}
    f = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2022, operator=FilterOperator.GTE),
            MetadataFilter(key="topic", value="pets"),
        ],
        condition=FilterCondition.AND,
    )
    assert query(store, "cat", filters=f).ids == ["n2"]
    f = MetadataFilters(
        filters=[
            MetadataFilter(key="topic", value=["code"], operator=FilterOperator.IN)
        ]
    )
    assert query(store, "cat", filters=f).ids == ["n3"]
    f = MetadataFilters(
        filters=[
            MetadataFilter(key="topic", value="x", operator=FilterOperator.TEXT_MATCH)
        ]
    )
    with pytest.raises(NotImplementedError):
        query(store, "cat", filters=f)


def test_node_ids_restrict_the_query(store):
    assert query(store, "rust", node_ids=["n1"]).ids == ["n1"]
    assert set(query(store, "rust", doc_ids=["doc1"]).ids) == {"n1", "n2"}
    assert len(query(store, "rust", node_ids=[]).ids) == 3


def test_delete_by_ref_doc_and_delete_nodes(store):
    store.delete("doc1")
    assert query(store, "cat").ids == ["n3"]
    store.delete_nodes(["n3"])
    assert query(store, "cat").ids == []


def test_node_without_embedding_is_rejected(store):
    with pytest.raises(ValueError):
        store.add([TextNode(id_="n9", text="no vector")])


def test_vector_store_index_end_to_end(tmp_path):
    from llama_index.core import Document, StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding

    class FakeEmbedding(BaseEmbedding):
        def _get_text_embedding(self, text):
            return embed(text)

        def _get_query_embedding(self, query):
            return embed(query)

        async def _aget_query_embedding(self, query):
            return embed(query)

    store = SrvDBVectorStore(srvdb.create(str(tmp_path), DIM))
    index = VectorStoreIndex.from_documents(
        [Document(text=t) for _, _, t, _ in ROWS],
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=FakeEmbedding(),
    )
    hits = index.as_retriever(similarity_top_k=1).retrieve("rust systems language")
    assert "rust" in hits[0].node.get_content()
