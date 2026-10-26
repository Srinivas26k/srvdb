---
title: FAQ
description: Common questions about SrvDB.
---

### What is SrvDB?
An open-source embedded vector database for Python and Rust. It stores
embeddings on local disk and finds the most similar ones to a query, inside
your own process: semantic search, RAG and AI memory without a server.

### Is SrvDB the same as SurrealDB?
No. SrvDB (S-R-V-D-B) is an embedded vector database library. SurrealDB is an
unrelated multi-model database server. Install SrvDB with `pip install srvdb`
or `cargo add srvdb`.

### Does SrvDB need a server or internet access?
No. It is a library inside your process and never opens a network connection.

### Which embedding models work?
Any model with a dimension between 128 and 4096. Use the model's own
dimension; no padding is needed.

### Do I have to normalise vectors?
No. SrvDB ranks by cosine similarity, which ignores length.

### Why don't my latest vectors survive a crash?
Only `persist()` makes writes durable. See [Durability](/srvdb/guides/durability/).

### Why do I get `BlockingIOError` when opening?
The database is already open, in this process or another. Share one
`Database` object between threads instead.

### Can I use SrvDB in a commercial product?
Yes. SrvDB is Apache-2.0 licensed. Commercial support is available:
srinivasvarma764@gmail.com.

### How fast is it?
We publish numbers only from reproducible benchmarks; see
[Benchmarks](/srvdb/project/benchmarks/).
