---
title: "Bug Dedup Engine"
description: "Links duplicate bug reports and similar stack traces. Level 3 Software engineering project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Bug Dedup Engine"
  order: 3
---

**Links duplicate bug reports and similar stack traces.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Software engineering](/srvdb/industries/software-engineering/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Bug Dedup Engine** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Duplicate bugs waste triage time.
- Stack traces vary slightly.
- Flaky tests hide patterns.

## How SrvDB powers Bug Dedup Engine

1. Bug texts and traces are embedded.
2. New bugs retrieve likely duplicates.
3. Flaky failures cluster.
4. Batch processing for CI runs.

## What changes

| Today | With Bug Dedup Engine on SrvDB |
|---|---|
| Manual dedup | Suggested duplicates |
| Scattered traces | Grouped traces |
| Flaky noise | Clustered flakes |

## How it works

1. Embed bugs and traces.
2. Match new ones.
3. Suggest duplicates.
4. Cluster flakes.
5. Close duplicates.

## SrvDB features it uses

- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Duplicate bugs
- Crash grouping
- Flaky tests
- Log anomalies
- Support escalations
- Release triage

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./bug-dedup-engine-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def likely_duplicates(bug_report, k=5):
    hits = db.search(embed(bug_report), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Triage leads confirm duplicates.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Software engineering projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Code Search](/srvdb/industries/software-engineering/code-search/) | Find code by what it does. |
| 2 · Intermediate | [Codebase Q&A](/srvdb/industries/software-engineering/codebase-qa/) | Ask how the system works and get answers citing files. |
| 3 · Advanced | **Bug Dedup Engine** (this page) | Links duplicate bug reports and similar stack traces. |
| 4 · Expert | [Coding Agent Memory](/srvdb/industries/software-engineering/coding-agent-memory/) | Long-term, crash-safe memory for autonomous coding agents. |
| 5 · Frontier | [Air-gapped Dev Copilot](/srvdb/industries/software-engineering/air-gapped-dev-copilot/) | A self-hosted developer assistant for isolated environments. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
