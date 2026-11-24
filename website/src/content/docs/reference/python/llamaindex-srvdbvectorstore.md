---
title: srvdb.integrations.llamaindex.SrvDBVectorStore
description: LlamaIndex vector store over a SrvDB database.
---

LlamaIndex vector store over a SrvDB database.

```python
class srvdb.integrations.llamaindex.SrvDBVectorStore(database, **kwargs)
```

## Parameters

- **database** : *database object*
  An open database whose dimension matches the node embeddings.

- ****kwargs** : *Any*
  Passed to LlamaIndex's `BasePydanticVectorStore`.

## Raises

- **ImportError**
  On import, when `llama-index-core` is not installed; install with `pip install "srvdb[llamaindex]"`.

- **ValueError**
  From `add`: a node has no embedding or the wrong dimension. From `query`: no `query_embedding`, or `similarity_top_k` below 1.

- **NotImplementedError**
  A filter uses an operator other than EQ, NE, GT, GTE, LT, LTE, IN, NIN.

- **OSError, RuntimeError**
  Passed through from the underlying database.

## Notes

- Methods: `add(nodes)`, `delete(ref_doc_id)`, `delete_nodes(node_ids=None, filters=None)`, `query(VectorStoreQuery)`. Property: `client` (the database object). `stores_text` is True.
- Nodes must carry embeddings; LlamaIndex computes them before `add`. Existing node ids are replaced.
- `MetadataFilters` support EQ, NE, GT, GTE, LT, LTE, IN and NIN with AND, OR and NOT. Filters, `node_ids` and `doc_ids` run inside SrvDB (see [Filtered search](/srvdb/guides/filtered-search/)), so a query returns `similarity_top_k` nodes whenever that many match; a key containing a dot falls back to filtering search results. A missing key matches only NE and NIN. More than 10,000 `node_ids` or `doc_ids` raise `ValueError`; so do an empty nested filter group, IN or NIN with a string value, and EQ or NE with a list value. `delete` by document id scans every record, O(n).
- Durability: every write calls `save()` before it returns.
- Thread safety: as thread-safe as the underlying database. The adapter makes no network calls.

## Examples

```python
import srvdb
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from srvdb.integrations.llamaindex import SrvDBVectorStore

v = [1.0] + [0.0] * 127
w = [0.0, 1.0] + [0.0] * 126
store = SrvDBVectorStore(srvdb.create("./vectors", dimension=128))
store.add([TextNode(id_="a", text="red apple", embedding=v), TextNode(id_="b", text="blue sky", embedding=w)])
result = store.query(VectorStoreQuery(query_embedding=v, similarity_top_k=1))
print(result.ids)
# ['a']
```

## See also

- [`srvdb.integrations.langchain.SrvDBVectorStore`](/srvdb/reference/python/langchain-srvdbvectorstore/): LangChain `VectorStore` over a SrvDB database.
- [`srvdb.open`](/srvdb/reference/python/open/): Open an existing database by path.
