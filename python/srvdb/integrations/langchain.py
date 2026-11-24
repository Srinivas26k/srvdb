"""LangChain ``VectorStore`` backed by an embedded SrvDB database.

Install with ``pip install "srvdb[langchain]"``.

>>> from srvdb.integrations.langchain import SrvDBVectorStore
>>> store = SrvDBVectorStore(srvdb.open("./vectors", dimension=384), embeddings)
>>> store.add_texts(texts, metadatas)
>>> store.similarity_search_with_score("query", k=4)

Scores are cosine similarities (higher is closer). Every write call
persists before it returns, so added and deleted documents survive a
crash. The store is as thread-safe as the underlying ``srvdb.Database``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Union

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "srvdb.integrations.langchain needs langchain-core. "
        'Install it with: pip install "srvdb[langchain]"'
    ) from exc

from srvdb import Database
from srvdb.integrations import _common

Filter = Union[dict[str, Any], Callable[[dict[str, Any]], bool]]


def _keep(filter: Filter | None) -> Callable[[str, dict[str, Any]], bool] | None:
    """Turn a LangChain filter (equality dict or predicate) into a test."""
    if filter is None:
        return None
    if callable(filter):
        pred = filter
        return lambda _id, rec: bool(pred(rec.get(_common.META_KEY, {})))
    if not isinstance(filter, dict):
        raise ValueError("filter must be a dict of metadata values or a callable")
    wanted = dict(filter)

    def keep(_id: str, rec: dict[str, Any]) -> bool:
        meta = rec.get(_common.META_KEY, {})
        return all(k in meta and meta[k] == v for k, v in wanted.items())

    return keep


def _doc(id_: str, record: dict[str, Any]) -> Document:
    return Document(
        id=id_,
        page_content=record.get(_common.TEXT_KEY, ""),
        metadata=record.get(_common.META_KEY, {}),
    )


class SrvDBVectorStore(VectorStore):
    """LangChain vector store over a ``srvdb.Database``.

    Args:
        database: an open ``srvdb.Database`` whose dimension matches the
            embedding model.
        embedding: the LangChain ``Embeddings`` used for texts and queries.

    Filters: ``filter`` is either a dict (every key must equal the stored
    metadata value) or a callable taking the metadata dict. Filtering runs
    on the search results, over-fetching until ``k`` matches are found.
    """

    def __init__(self, database: Database, embedding: Embeddings) -> None:
        self._db = database
        self._embedding = embedding

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def database(self) -> Database:
        """The underlying ``srvdb.Database``."""
        return self._db

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and store ``texts``; existing ids are replaced.

        Raises ValueError if lengths differ or the embedding dimension does
        not match the database.
        """
        texts = list(texts)
        metadatas = metadatas if metadatas is not None else [{} for _ in texts]
        ids = list(ids) if ids is not None else _common.new_ids(len(texts))
        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError("texts, metadatas and ids must have the same length")
        if not texts:
            return []
        vectors = self._embedding.embed_documents(texts)
        records = (
            {_common.TEXT_KEY: t, _common.META_KEY: dict(m or {})}
            for t, m in zip(texts, metadatas)
        )
        _common.write(self._db, ids, vectors, records)
        return ids

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Store ``documents``, keeping their ``id`` when one is set."""
        ids = kwargs.pop("ids", None)
        if ids is None and all(d.id for d in documents):
            ids = [d.id for d in documents]
        return self.add_texts(
            [d.page_content for d in documents],
            [d.metadata for d in documents],
            ids=ids,
            **kwargs,
        )

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete ``ids``; unknown ids are ignored. Requires ``ids``."""
        if ids is None:
            raise ValueError("delete needs a list of ids")
        _common.remove(self._db, ids)
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Return stored documents for ``ids``; unknown ids are skipped."""
        docs = []
        for id_ in ids:
            record = _common.load(self._db, id_)
            if record is not None:
                docs.append(_doc(id_, record))
        return docs

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, filter: Filter | None = None
    ) -> list[tuple[Document, float]]:
        """Return ``(document, cosine score)`` pairs, best first."""
        hits = _common.search(self._db, embedding, k, _keep(filter))
        return [(_doc(id_, rec), score) for id_, score, rec in hits]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Filter | None = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        vector = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(vector, k, filter)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Filter | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        pairs = self.similarity_search_with_score_by_vector(embedding, k, filter)
        return [doc for doc, _ in pairs]

    def similarity_search(
        self, query: str, k: int = 4, filter: Filter | None = None, **kwargs: Any
    ) -> list[Document]:
        pairs = self.similarity_search_with_score(query, k, filter)
        return [doc for doc, _ in pairs]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # Cosine similarity in [-1, 1] mapped to a relevance in [0, 1].
        return lambda score: (score + 1.0) / 2.0

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        path: str | None = None,
        mode: str = "flat",
        **kwargs: Any,
    ) -> SrvDBVectorStore:
        """Create (or open) a database at ``path`` and add ``texts``.

        The dimension is taken from the embedding of the first text.
        Raises ValueError if ``path`` is missing or ``texts`` is empty.
        """
        if path is None:
            raise ValueError("from_texts needs path= for the database directory")
        if not texts:
            raise ValueError(
                "from_texts needs at least one text to learn the dimension"
            )
        dimension = len(embedding.embed_query(texts[0]))
        store = cls(Database(path, dimension=dimension, mode=mode), embedding)
        store.add_texts(texts, metadatas, ids=ids)
        return store
