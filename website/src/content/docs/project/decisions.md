---
title: Design decisions
description: Architecture decision records (ADRs).
---

Significant decisions are written down as ADRs: the problem, the decision,
the alternatives and the consequences.

| ADR | Decision |
|-----|----------|
| [0001](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0001-workspace-split.md) | Split into a pure Rust core and thin Python bindings |
| [0002](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0002-typed-errors.md) | Typed errors in Rust, specific exceptions in Python |
| [0003](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0003-durability.md) | Durability guarantee and commit protocol |
| [0004](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0004-delete-update-compact.md) | Delete, update and compaction |
| [0005](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0005-concurrency.md) | Concurrency: one open instance, thread-safe handle |
| [0006](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0006-apache-license.md) | Relicense under Apache-2.0 |
| [0013](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0013-simple-python-verbs.md) | Python verbs: create, open, save, close, delete |
| [0014](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0014-python-filtered-search.md) | Declarative metadata filters (`where=`) in the core and Python |
| [0015](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0015-keyword-search.md) | BM25 keyword search and hybrid ranking (Reciprocal Rank Fusion) |
| [0018](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0018-memory-budget.md) | Memory budget: pick flat or 4/2/1-bit RaBitQ to fit `memory_mb` |
