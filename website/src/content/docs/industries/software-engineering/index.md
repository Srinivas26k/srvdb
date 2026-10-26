---
title: "Software engineering"
description: "Five SrvDB projects for software engineering, from beginner to frontier."
sidebar:
  label: Overview
  order: 0
---

Developers want AI help with their codebase without uploading proprietary code.

## The challenge

- Finding "where do we validate tokens?" by keyword is slow.
- Internal docs and code drift apart.
- Many companies forbid sending source code to external services.

## Why SrvDB fits

- **Code search by meaning** with a code embedding model.
- **Keep the index current** with `upsert` and `delete` as files change.
- **Everything local**, including in CI.

## Five projects, five levels

Each project has its own page: the problem, how SrvDB powers it, what changes, architecture, use cases, core code and limits.

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Code Search](/srvdb/industries/software-engineering/code-search/) | Find code by what it does. |
| 2 · Intermediate | [Codebase Q&A](/srvdb/industries/software-engineering/codebase-qa/) | Ask how the system works and get answers citing files. |
| 3 · Advanced | [Bug Dedup Engine](/srvdb/industries/software-engineering/bug-dedup-engine/) | Links duplicate bug reports and similar stack traces. |
| 4 · Expert | [Coding Agent Memory](/srvdb/industries/software-engineering/coding-agent-memory/) | Long-term, crash-safe memory for autonomous coding agents. |
| 5 · Frontier | [Air-gapped Dev Copilot](/srvdb/industries/software-engineering/air-gapped-dev-copilot/) | A self-hosted developer assistant for isolated environments. |

:::note[Coming soon]
Runnable reference implementations of these projects are in progress.
:::
