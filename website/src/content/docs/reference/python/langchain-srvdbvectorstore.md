---
title: srvdb.integrations.langchain.SrvDBVectorStore
description: LangChain VectorStore over a srvdb.Database.
---

LangChain `VectorStore` over a `srvdb.Database`.

```python
class srvdb.integrations.langchain.SrvDBVectorStore(database: srvdb.Database, embedding: Embeddings)
```

## Parameters

- **database** : *srvdb.Database*
  An open database whose dimension matches the embedding model.

- **embedding** : *langchain_core.embeddings.Embeddings*
  Model used for texts and queries.

## Raises

- **ImportError**
  On import, when `langchain-core` is not installed; install with `pip install "srvdb[langchain]"`.

- **ValueError**
  From `add_texts`: lengths differ or an embedding has the wrong dimension. From searches: `k` below 1, a `filter` that is neither a dict nor a callable. From `delete`: `ids` is None. From `from_texts`: `path` missing or `texts` empty.

- **OSError, RuntimeError**
  Passed through from the underlying `Database`.

## Notes

- Methods: `add_texts(texts, metadatas=None, *, ids=None)`, `add_documents(documents)`, `delete(ids)`, `get_by_ids(ids)`, `similarity_search(query, k=4, filter=None)`, `similarity_search_with_score(...)`, `similarity_search_by_vector(...)`, `similarity_search_with_score_by_vector(...)`, and the classmethod `from_texts(texts, embedding, metadatas=None, *, ids=None, path, mode="flat")`. Properties: `embeddings`, `database`.
- Existing ids are replaced (writes use `upsert`). Without `ids`, random UUIDs are assigned.
- `filter` is a dict (every key must equal the stored metadata value) or a callable taking the metadata dict. Filtering runs on search results, over-fetching until `k` matches are found.
- Scores are cosine similarities; LangChain relevance scores map them to `(score + 1) / 2`.
- Durability: every write method calls `persist()` before it returns, so added and deleted documents survive a crash.
- Thread safety: as thread-safe as the underlying `Database`. The adapter makes no network calls.

## Examples

```python
import srvdb
from langchain_core.embeddings import DeterministicFakeEmbedding
from srvdb.integrations.langchain import SrvDBVectorStore

store = SrvDBVectorStore(srvdb.open("./vectors", dimension=128), DeterministicFakeEmbedding(size=128))
store.add_texts(["red apple", "blue sky"], [{"kind": "fruit"}, {"kind": "weather"}])
docs = store.similarity_search("red apple", k=1, filter={"kind": "fruit"})
print(docs[0].page_content)
# red apple
```

## See also

- [`srvdb.integrations.llamaindex.SrvDBVectorStore`](/srvdb/reference/python/llamaindex-srvdbvectorstore/): LlamaIndex vector store over a `srvdb.Database`.
- [`srvdb.open`](/srvdb/reference/python/open/): Open a database, creating it if it does not exist.
