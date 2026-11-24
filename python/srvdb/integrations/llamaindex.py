"""LlamaIndex vector store backed by an embedded SrvDB database.

Install with ``pip install "srvdb[llamaindex]"``.

>>> from srvdb.integrations.llamaindex import SrvDBVectorStore
>>> store = SrvDBVectorStore(srvdb.open("./vectors", dimension=384))
>>> ctx = StorageContext.from_defaults(vector_store=store)

Nodes must carry embeddings (LlamaIndex computes them before ``add``).
Scores are cosine similarities. Every write persists before it returns.
"""

from __future__ import annotations

import operator
from typing import Any

try:
    from llama_index.core.schema import BaseNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.vector_stores.utils import (
        metadata_dict_to_node,
        node_to_metadata_dict,
    )
    from pydantic import PrivateAttr
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "srvdb.integrations.llamaindex needs llama-index-core. "
        'Install it with: pip install "srvdb[llamaindex]"'
    ) from exc

from srvdb import Database
from srvdb.integrations import _common

_OPS = {
    FilterOperator.EQ: operator.eq,
    FilterOperator.NE: operator.ne,
    FilterOperator.GT: operator.gt,
    FilterOperator.GTE: operator.ge,
    FilterOperator.LT: operator.lt,
    FilterOperator.LTE: operator.le,
    FilterOperator.IN: lambda a, b: a in b,
    FilterOperator.NIN: lambda a, b: a not in b,
}


def _match(meta: dict[str, Any], filters: MetadataFilters) -> bool:
    results = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            results.append(_match(meta, f))
            continue
        assert isinstance(f, MetadataFilter)
        op = _OPS.get(f.operator)
        if op is None:
            raise NotImplementedError(
                f"SrvDB does not support the {f.operator} filter operator; "
                f"supported: {', '.join(o.value for o in _OPS)}"
            )
        if f.key not in meta:
            results.append(f.operator == FilterOperator.NE)
            continue
        try:
            results.append(bool(op(meta[f.key], f.value)))
        except TypeError:  # incomparable types simply do not match
            results.append(False)
    if filters.condition == FilterCondition.OR:
        return any(results)
    if filters.condition == FilterCondition.NOT:
        return not any(results)
    return all(results)


class SrvDBVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store over a ``srvdb.Database``.

    Supports ``add``, ``delete`` (by ``ref_doc_id``), ``delete_nodes`` and
    default (dense) queries with ``MetadataFilters`` (EQ, NE, GT, GTE, LT,
    LTE, IN, NIN; AND/OR/NOT). Filters run on search results, over-fetching
    until ``similarity_top_k`` matches are found. ``delete`` by document id
    scans all stored records, O(n).
    """

    stores_text: bool = True
    flat_metadata: bool = False
    _db: Database = PrivateAttr()

    def __init__(self, database: Database, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._db = database

    @classmethod
    def class_name(cls) -> str:
        """Name LlamaIndex uses when serialising the store."""
        return "SrvDBVectorStore"

    @property
    def client(self) -> Database:
        """The underlying ``srvdb.Database``."""
        return self._db

    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        """Store nodes with their embeddings; ValueError if one has none."""
        ids, vectors, records = [], [], []
        for node in nodes:
            embedding = node.get_embedding()  # raises ValueError if missing
            ids.append(node.node_id)
            vectors.append(embedding)
            records.append(
                {
                    _common.TEXT_KEY: node.get_content(),
                    _common.META_KEY: node_to_metadata_dict(
                        node, remove_text=False, flat_metadata=self.flat_metadata
                    ),
                }
            )
        _common.write(self._db, ids, vectors, records)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete every node whose source document is ``ref_doc_id``."""
        doomed = []
        for id_ in self._db.get_all_ids():
            record = _common.load(self._db, id_)
            meta = (record or {}).get(_common.META_KEY, {})
            if meta.get("ref_doc_id") == ref_doc_id or meta.get("doc_id") == ref_doc_id:
                doomed.append(id_)
        _common.remove(self._db, doomed)

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by id and/or by metadata filter."""
        if filters is None:
            _common.remove(self._db, list(node_ids or []))
            return
        candidates = node_ids if node_ids is not None else self._db.get_all_ids()
        doomed = []
        for id_ in candidates:
            record = _common.load(self._db, id_)
            if record is not None and _match(record.get(_common.META_KEY, {}), filters):
                doomed.append(id_)
        _common.remove(self._db, doomed)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Dense top-k search; ValueError if the query has no embedding.

        Raises NotImplementedError for an unsupported filter operator.
        """
        if query.query_embedding is None:
            raise ValueError("SrvDBVectorStore needs query.query_embedding")
        filters = query.filters
        # LlamaIndex passes an empty list to mean "no restriction".
        allowed = set(query.node_ids) if query.node_ids else None
        docs = set(query.doc_ids) if query.doc_ids else None

        def keep(id_: str, rec: dict[str, Any]) -> bool:
            meta = rec.get(_common.META_KEY, {})
            if allowed is not None and id_ not in allowed:
                return False
            if docs is not None and meta.get("ref_doc_id") not in docs:
                return False
            return filters is None or _match(meta, filters)

        use_keep = filters is not None or allowed is not None or docs is not None
        hits = _common.search(
            self._db,
            query.query_embedding,
            query.similarity_top_k,
            keep if use_keep else None,
        )
        nodes, sims, ids = [], [], []
        for id_, score, record in hits:
            node = metadata_dict_to_node(record.get(_common.META_KEY, {}))
            node.set_content(record.get(_common.TEXT_KEY, ""))
            nodes.append(node)
            sims.append(score)
            ids.append(id_)
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)
