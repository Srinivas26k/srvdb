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
