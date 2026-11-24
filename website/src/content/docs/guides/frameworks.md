---
title: LangChain and LlamaIndex
description: Use SrvDB as the vector store in LangChain or LlamaIndex, fully offline.
---

SrvDB ships optional vector-store adapters for LangChain and LlamaIndex.
They are plain Python over a SrvDB database; the frameworks are extras, so
`import srvdb` never imports them.

```bash
pip install "srvdb[langchain]"     # langchain-core
pip install "srvdb[llamaindex]"    # llama-index-core
```

Importing an adapter without its extra raises an `ImportError` that names
the extra to install.

Both adapters behave the same way:

- **Scores** are cosine similarities, best first.
- **Writes persist.** Every add or delete calls `save()` before it
  returns, so it survives a crash.
- **Metadata filters** run on the search results: the adapter fetches
  more candidates until it has `k` matches or has seen every vector.
  Very selective filters on large databases therefore cost close to a
  full scan.
- **No network.** The adapters only call the embedding model you pass in.
  The examples below use a small hash-based embedding so they run with no
  model download; swap in a real local model (for example a
  sentence-transformers model you have already downloaded) for real use.

The examples need a dimension of at least 128 and a fresh directory.

## LangChain

```python
import hashlib
import tempfile

from langchain_core.embeddings import Embeddings

import srvdb
from srvdb.integrations.langchain import SrvDBVectorStore

DIM = 384


class HashEmbeddings(Embeddings):
    """Toy offline embedding: hashes words into buckets."""

    def embed_query(self, text):
        vec = [0.0] * DIM
        for word in text.lower().split():
            vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


db = srvdb.create(tempfile.mkdtemp(), dimension=DIM)
store = SrvDBVectorStore(db, HashEmbeddings())
store.add_texts(
    ["SrvDB runs inside your process", "Rust has no garbage collector"],
    metadatas=[{"topic": "db"}, {"topic": "lang"}],
    ids=["a", "b"],
)

for doc, score in store.similarity_search_with_score("where does srvdb run", k=1):
    print(doc.id, round(score, 3), doc.page_content)

# Filters: a dict of equal values, or a callable on the metadata dict.
print(store.similarity_search("srvdb", k=2, filter={"topic": "lang"}))

retriever = store.as_retriever(search_kwargs={"k": 1})
print(retriever.invoke("garbage collector"))

store.delete(["a"])
```

`SrvDBVectorStore.from_texts(texts, embedding, path="./vectors")` creates
the database for you, taking the dimension from the first embedding.

## LlamaIndex

```python
import hashlib
import tempfile

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

import srvdb
from srvdb.integrations.llamaindex import SrvDBVectorStore

DIM = 384


def hash_embed(text):
    vec = [0.0] * DIM
    for word in text.lower().split():
        vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM] += 1.0
    return vec


class HashEmbedding(BaseEmbedding):
    """Toy offline embedding: hashes words into buckets."""

    def _get_text_embedding(self, text):
        return hash_embed(text)

    def _get_query_embedding(self, query):
        return hash_embed(query)

    async def _aget_query_embedding(self, query):
        return hash_embed(query)


store = SrvDBVectorStore(srvdb.create(tempfile.mkdtemp(), dimension=DIM))
index = VectorStoreIndex.from_documents(
    [
        Document(text="SrvDB runs inside your process", metadata={"topic": "db"}),
        Document(text="Rust has no garbage collector", metadata={"topic": "lang"}),
    ],
    storage_context=StorageContext.from_defaults(vector_store=store),
    embed_model=HashEmbedding(),
)

for hit in index.as_retriever(similarity_top_k=1).retrieve("where does srvdb run"):
    print(round(hit.score, 3), hit.node.get_content())

only_lang = MetadataFilters(filters=[MetadataFilter(key="topic", value="lang")])
print(index.as_retriever(filters=only_lang).retrieve("srvdb"))
```

Supported filter operators are `==`, `!=`, `>`, `>=`, `<`, `<=`, `in` and
`nin`, combined with `and`, `or` or `not`; other operators raise
`NotImplementedError`. `store.delete(ref_doc_id)` removes every node of a
source document (it scans all stored records), and `store.delete_nodes`
removes nodes by id or filter.
