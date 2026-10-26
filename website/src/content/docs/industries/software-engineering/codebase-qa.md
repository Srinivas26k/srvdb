---
title: "Codebase Q&A"
description: "Ask how the system works and get answers citing files. Level 2 Software engineering project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Codebase Q&A"
  order: 2
---

**Ask how the system works and get answers citing files.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Software engineering](/srvdb/industries/software-engineering/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Codebase Q&A** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Architecture knowledge is undocumented.
- New developers ask the same questions.
- Docs drift from code.

## How SrvDB powers Codebase Q&A

1. Code, ADRs and docs are embedded.
2. Answers cite files and lines.
3. Local LLM keeps code private.
4. Indexed on every merge.

## What changes

| Today | With Codebase Q&A on SrvDB |
|---|---|
| Asking seniors | Cited answers |
| Stale docs | Answers from current code |
| Private code risk | Local stack |

## How it works

1. Index repo and docs.
2. Ask questions.
3. Verify citations.
4. Re-index on merge.
5. Track gaps.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Onboarding
- Design questions
- Incident debugging
- API usage
- Review preparation
- Documentation gaps

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./codebase-qa-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def ask_codebase(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Verify answers against cited code.

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
| 2 · Intermediate | **Codebase Q&A** (this page) | Ask how the system works and get answers citing files. |
| 3 · Advanced | [Bug Dedup Engine](/srvdb/industries/software-engineering/bug-dedup-engine/) | Links duplicate bug reports and similar stack traces. |
| 4 · Expert | [Coding Agent Memory](/srvdb/industries/software-engineering/coding-agent-memory/) | Long-term, crash-safe memory for autonomous coding agents. |
| 5 · Frontier | [Air-gapped Dev Copilot](/srvdb/industries/software-engineering/air-gapped-dev-copilot/) | A self-hosted developer assistant for isolated environments. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
