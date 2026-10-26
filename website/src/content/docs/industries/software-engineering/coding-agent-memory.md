---
title: "Coding Agent Memory"
description: "Long-term, crash-safe memory for autonomous coding agents. Level 4 Software engineering project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Coding Agent Memory"
  order: 4
---

**Long-term, crash-safe memory for autonomous coding agents.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Software engineering](/srvdb/industries/software-engineering/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Coding Agent Memory** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Agents forget decisions between sessions.
- Long runs crash and lose context.
- Repos change constantly.

## How SrvDB powers Coding Agent Memory

1. Decisions, attempts and outcomes are stored in SrvDB.
2. Agents recall relevant memory per task.
3. `persist()` checkpoints survive crashes.
4. Per-repo databases.

## What changes

| Today | With Coding Agent Memory on SrvDB |
|---|---|
| Agents repeat mistakes | Memory of outcomes |
| Lost context on crash | Crash-safe checkpoints |
| Global confusion | Per-repo memory |

## How it works

1. Log agent steps.
2. Embed summaries.
3. Recall per task.
4. Persist checkpoints.
5. Prune with delete.

## SrvDB features it uses

- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.

## Use cases

- Autonomous refactors
- Long migrations
- Test fixing
- Dependency upgrades
- Code review agents
- Incident agents

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./coding-agent-memory-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def recall_memory(task, k=5):
    hits = db.search(embed(task), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Review agent changes before merging.

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
| 3 · Advanced | [Bug Dedup Engine](/srvdb/industries/software-engineering/bug-dedup-engine/) | Links duplicate bug reports and similar stack traces. |
| 4 · Expert | **Coding Agent Memory** (this page) | Long-term, crash-safe memory for autonomous coding agents. |
| 5 · Frontier | [Air-gapped Dev Copilot](/srvdb/industries/software-engineering/air-gapped-dev-copilot/) | A self-hosted developer assistant for isolated environments. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
