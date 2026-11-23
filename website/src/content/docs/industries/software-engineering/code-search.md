---
title: "Code Search"
description: "Find code by what it does. Level 1 Software engineering project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Code Search"
  order: 1
---

**Find code by what it does.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Software engineering](/srvdb/industries/software-engineering/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Code Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Naming conventions hide functionality.
- Large repos are hard to navigate.
- Proprietary code can't leave.

## How SrvDB powers Code Search

1. Functions and doc sections are embedded with a code model.
2. Questions find matching code.
3. Runs locally or in CI.
4. Changes indexed with `upsert` and `delete`.

## What changes

| Today | With Code Search on SrvDB |
|---|---|
| grep for names | Search by behaviour |
| Tribal knowledge | Searchable codebase |
| Cloud indexing | Local index |

## How it works

1. Parse repo.
2. Embed chunks.
3. Search.
4. Update on commit.
5. Share the index.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- "Where do we validate tokens?"
- Find examples of an API
- Locate error handling
- Find config loaders
- Onboarding
- Refactoring scope

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./code-search-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def where_is(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Embedding model choice matters for code.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Software engineering projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Code Search** (this page) | Find code by what it does. |
| 2 · Intermediate | [Codebase Q&A](/srvdb/industries/software-engineering/codebase-qa/) | Ask how the system works and get answers citing files. |
| 3 · Advanced | [Bug Dedup Engine](/srvdb/industries/software-engineering/bug-dedup-engine/) | Links duplicate bug reports and similar stack traces. |
| 4 · Expert | [Coding Agent Memory](/srvdb/industries/software-engineering/coding-agent-memory/) | Long-term, crash-safe memory for autonomous coding agents. |
| 5 · Frontier | [Air-gapped Dev Copilot](/srvdb/industries/software-engineering/air-gapped-dev-copilot/) | A self-hosted developer assistant for isolated environments. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
