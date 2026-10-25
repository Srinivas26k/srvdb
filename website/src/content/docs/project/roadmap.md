---
title: Roadmap
description: What is planned after 0.3.
---

Plans, not promises. Each item ships only with tests, docs and, for
performance work, a reproducible benchmark.

## 0.3: Trustworthy storage ✅

Crash-safe `persist()`, delete/upsert/compact, validated input, typed errors,
thread safety, correctness fixes, Apache-2.0.

## 0.4: Retrieval

- Metadata filtering in search
- Hybrid (keyword + vector) search
- LangChain and LlamaIndex integrations
- Research: training-free quantization (TurboQuant-style) to replace PQ's
  accuracy loss on clustered data

## Later

- Reproducible public benchmark suite against other local vector stores
- crates.io release and docs.rs API docs
- Encryption at rest

Ideas and votes welcome in
[GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions).
