---
title: Python API reference
description: Every public object of the srvdb Python package, one page each.
---

This section documents every public object of the `srvdb` package. Each page gives the signature, parameters, return value, the exact exceptions raised, durability and thread-safety notes, and a runnable example. Type stubs ship with the package, so editors show the same signatures.

Errors follow one rule: bad input raises `ValueError`, operating-system I/O failures raise `OSError` (`BlockingIOError` when the database is already open), and internal faults or damaged data raise `RuntimeError`. See [Handling errors](/srvdb/guides/errors/).

## Top level

| Object | Summary |
|--------|---------|
| [`srvdb.open`](/srvdb/reference/python/open/) | Open a database, creating it if it does not exist. |

## Database class and constructors

| Object | Summary |
|--------|---------|
| [`srvdb.Database`](/srvdb/reference/python/database/) | An open SrvDB database. |
| [`srvdb.Database.new_hnsw`](/srvdb/reference/python/database-new-hnsw/) | Open or create an HNSW database with explicit graph parameters. |
| [`srvdb.Database.new_scalar_quantized`](/srvdb/reference/python/database-new-scalar-quantized/) | Open or create an SQ8 database trained on a sample you choose. |
| [`srvdb.Database.new_product_quantized`](/srvdb/reference/python/database-new-product-quantized/) | Open or create a PQ database trained on a sample you choose. |

## Writing

| Object | Summary |
|--------|---------|
| [`srvdb.Database.add`](/srvdb/reference/python/database-add/) | Add new vectors with their ids and metadata. |
| [`srvdb.Database.upsert`](/srvdb/reference/python/database-upsert/) | Add vectors, replacing any that already have the same id. |
| [`srvdb.Database.delete`](/srvdb/reference/python/database-delete/) | Delete vectors by id. |
| [`srvdb.Database.persist`](/srvdb/reference/python/database-persist/) | Make every earlier write durable (crash-safe commit). |
| [`srvdb.Database.compact`](/srvdb/reference/python/database-compact/) | Remove deleted and replaced vectors from disk. |

## Reading

| Object | Summary |
|--------|---------|
| [`srvdb.Database.search`](/srvdb/reference/python/database-search/) | Find the k nearest vectors to a query. |
| [`srvdb.Database.search_batch`](/srvdb/reference/python/database-search-batch/) | Run several searches in one call. |
| [`srvdb.Database.get`](/srvdb/reference/python/database-get/) | Return the metadata stored for an id. |
| [`srvdb.Database.get_all_ids`](/srvdb/reference/python/database-get-all-ids/) | Return every live id. |
| [`srvdb.Database.count`](/srvdb/reference/python/database-count/) | Return the number of searchable vectors. |
| [`srvdb.Database.info`](/srvdb/reference/python/database-info/) | Return a one-line summary of the database. |

## Tuning

| Object | Summary |
|--------|---------|
| [`srvdb.Database.set_mode`](/srvdb/reference/python/database-set-mode/) | Switch the index mode of an open database. |
| [`srvdb.Database.set_ef_search`](/srvdb/reference/python/database-set-ef-search/) | Set HNSW's search candidate list size. |
| [`srvdb.Database.configure_ivf`](/srvdb/reference/python/database-configure-ivf/) | Set the IVF partition parameters. |
| [`srvdb.Database.train_ivf`](/srvdb/reference/python/database-train-ivf/) | Train the IVF index on the stored vectors. |

## Framework adapters

| Object | Summary |
|--------|---------|
| [`srvdb.integrations.langchain.SrvDBVectorStore`](/srvdb/reference/python/langchain-srvdbvectorstore/) | LangChain `VectorStore` over a `srvdb.Database`. |
| [`srvdb.integrations.llamaindex.SrvDBVectorStore`](/srvdb/reference/python/llamaindex-srvdbvectorstore/) | LlamaIndex vector store over a `srvdb.Database`. |

## Constants

| Object | Summary |
|--------|---------|
| `srvdb.__version__` | The installed version string, for example `"0.3.0"`. |

---

Every example in this section is checked by a script. To run it, build the extension with `maturin develop` (with `langchain-core` and `llama-index-core` installed for the adapter pages), then run `python website/scripts/check_examples.py` from the repository root. Each example runs in its own temporary directory.
