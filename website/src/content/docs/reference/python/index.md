---
title: Python API reference
description: Every public object of the srvdb Python package, one page each.
---

This section documents every public object of the `srvdb` package. Each page gives the signature, parameters, return value, the exact exceptions raised, durability and thread-safety notes, and a runnable example. Type stubs ship with the package, so editors show the same signatures.

Errors follow one rule: bad input raises `ValueError`, operating-system I/O failures raise `OSError` (`FileNotFoundError` when there is no database, `FileExistsError` when `create` finds one, `BlockingIOError` when it is already open), and internal faults or damaged data raise `RuntimeError`. See [Handling errors](/srvdb/guides/errors/).

## Top level

| Object | Summary |
|--------|---------|
| [`srvdb.create`](/srvdb/reference/python/create/) | Create a new database with its index and options. |
| [`srvdb.open`](/srvdb/reference/python/open/) | Open an existing database by path. |
| [`srvdb.delete`](/srvdb/reference/python/delete/) | Delete a database from disk. |
| [The database object](/srvdb/reference/python/db/) | What `create` and `open` return (`db`), and the `with` block. |

## Writing

| Object | Summary |
|--------|---------|
| [`db.add`](/srvdb/reference/python/db-add/) | Add new vectors with their ids and metadata. |
| [`db.upsert`](/srvdb/reference/python/db-upsert/) | Add vectors, replacing any that already have the same id. |
| [`db.delete`](/srvdb/reference/python/db-delete/) | Delete vectors by id. |
| [`db.save`](/srvdb/reference/python/db-save/) | Make every earlier write durable (crash-safe commit); `db.persist()` is an alias. |
| [`db.close`](/srvdb/reference/python/db-close/) | Release the database and its lock without saving. |
| [`db.compact`](/srvdb/reference/python/db-compact/) | Remove deleted and replaced vectors from disk. |

## Reading

| Object | Summary |
|--------|---------|
| [`db.search`](/srvdb/reference/python/db-search/) | Find the k nearest vectors to a query. |
| [`db.search_batch`](/srvdb/reference/python/db-search-batch/) | Run several searches in one call. |
| [`db.get`](/srvdb/reference/python/db-get/) | Return the metadata stored for an id. |
| [`db.get_all_ids`](/srvdb/reference/python/db-get-all-ids/) | Return every live id. |
| [`db.count`](/srvdb/reference/python/db-count/) | Return the number of searchable vectors. |
| [`db.info`](/srvdb/reference/python/db-info/) | Return a one-line summary of the database. |

## Tuning

| Object | Summary |
|--------|---------|
| [`db.set_rabitq_rerank`](/srvdb/reference/python/db-set-rabitq-rerank/) | Change the rerank factor of a RaBitQ database. |
| [`db.set_mode`](/srvdb/reference/python/db-set-mode/) | Switch the index mode of an open database. |
| [`db.set_ef_search`](/srvdb/reference/python/db-set-ef-search/) | Set HNSW's search candidate list size. |
| [`db.configure_ivf`](/srvdb/reference/python/db-configure-ivf/) | Set the IVF partition parameters. |
| [`db.train_ivf`](/srvdb/reference/python/db-train-ivf/) | Train the IVF index on the stored vectors. |

## Framework adapters

| Object | Summary |
|--------|---------|
| [`srvdb.integrations.langchain.SrvDBVectorStore`](/srvdb/reference/python/langchain-srvdbvectorstore/) | LangChain `VectorStore` over a SrvDB database. |
| [`srvdb.integrations.llamaindex.SrvDBVectorStore`](/srvdb/reference/python/llamaindex-srvdbvectorstore/) | LlamaIndex vector store over a SrvDB database. |

## Constants

| Object | Summary |
|--------|---------|
| `srvdb.__version__` | The installed version string, for example `"0.3.0"`. |

---

Every example in this section is checked by a script. To run it, build the extension with `maturin develop` (with `langchain-core` and `llama-index-core` installed for the adapter pages), then run `python website/scripts/check_examples.py` from the repository root. Each example runs in its own temporary directory.
