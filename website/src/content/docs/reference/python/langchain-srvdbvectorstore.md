---
title: srvdb.integrations.langchain.SrvDBVectorStore
description: LangChain VectorStore over a SrvDB database.
---

LangChain `VectorStore` over a SrvDB database.

```python
class srvdb.integrations.langchain.SrvDBVectorStore(database, embedding: Embeddings)
```

## Parameters

- **database** : *database object*
  An open database whose dimension matches the embedding model.

- **embedding** : *langchain_core.embeddings.Embeddings*
  Model used for texts and queries.

## Raises

- **ImportError**
  On import, when `langchain-core` is not installed; install with `pip install "srvdb[langchain]"`.

- **ValueError**
  From `add_texts`: lengths differ or an embedding has the wrong dimension. From searches: `k` below 1, a `filter` that is neither a dict nor a callable, or a dict filter with an unknown operator or wrong operand type. From `delete`: `ids` is None. From `from_texts`: `path` missing or `texts` empty.

- **OSError, RuntimeError**
  Passed through from the underlying database.

## Notes

- Methods: `add_texts(texts, metadatas=None, *, ids=None)`, `add_documents(documents)`, `delete(ids)`, `get_by_ids(ids)`, `similarity_search(query, k=4, filter=None)`, `similarity_search_with_score(...)`, `similarity_search_by_vector(...)`, `similarity_search_with_score_by_vector(...)`, and the classmethod `from_texts(texts, embedding, metadatas=None, *, ids=None, path, index="hnsw", **options)` (opens an existing database with the same settings, see [`srvdb.create`](/srvdb/reference/python/create/) with `exist_ok=True`). Properties: `embeddings`, `database`.
- Existing ids are replaced (writes use `upsert`). Without `ids`, random UUIDs are assigned.
- `filter` is a dict or a callable taking the metadata dict. A dict runs inside SrvDB (see [Filtered search](/srvdb/guides/filtered-search/)): `{"kind": "fruit"}` for equality, or operators such as `{"price": {"$lt": 5}}` and `$and` / `$or` / `$not`, and returns `k` documents whenever `k` match. A callable runs on search results, over-fetching until `k` matches are found.
- Scores are cosine similarities; LangChain relevance scores map them to `(score + 1) / 2`.
- Durability: every write method calls `save()` before it returns, so added and deleted documents survive a crash.
- Thread safety: as thread-safe as the underlying database. The adapter makes no network calls.

## Examples

```python
import srvdb
from langchain_core.embeddings import DeterministicFakeEmbedding
from srvdb.integrations.langchain import SrvDBVectorStore

store = SrvDBVectorStore(srvdb.create("./vectors", dimension=128), DeterministicFakeEmbedding(size=128))
store.add_texts(["red apple", "blue sky"], [{"kind": "fruit"}, {"kind": "weather"}])
docs = store.similarity_search("red apple", k=1, filter={"kind": "fruit"})
print(docs[0].page_content)
# red apple
```

## See also

- [`srvdb.integrations.llamaindex.SrvDBVectorStore`](/srvdb/reference/python/llamaindex-srvdbvectorstore/): LlamaIndex vector store over a SrvDB database.
- [`srvdb.open`](/srvdb/reference/python/open/): Open an existing database by path.
