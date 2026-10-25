---
title: Introduction
description: What SrvDB is, who it is for, and what it promises.
---

SrvDB is an **embedded vector database**: a library you import, which stores
embeddings on local disk and finds the most similar ones to a query. It is
written in Rust and used from Python or Rust.

## Who it is for

- **Local and offline AI apps**: RAG over private documents, assistants that
  run on a laptop, tools for air-gapped environments.
- **Desktop and edge software** that needs semantic search without shipping a
  database server.
- **Developers who want it to just work**: `pip install srvdb`, no Docker, no
  configuration.

## What it promises

1. **Your committed data is safe.** `persist()` is a real commit point.
   After it returns, data survives a crash or power loss. After a crash, the
   database reopens in exactly the state of the last `persist()`. See
   [Durability](/srvdb/guides/durability/).
2. **Results mean what they say.** Scores are true cosine similarities in
   [-1, 1], in every mode.
3. **Bad input is rejected, not stored.** Wrong dimensions, NaN, duplicate ids
   and invalid metadata raise clear errors before anything is written.
4. **No surprises.** SrvDB never changes how your data is stored unless you
   ask, never opens a network connection, and never prints to your console.

## What it is not

SrvDB is a single-node library. It has no server, replication or sharding;
scaling across machines is your application's job. One database directory
can be open in one place at a time.

## Next

- [Install SrvDB](/srvdb/start/installation/)
- [Quickstart in 5 minutes](/srvdb/start/quickstart/)
