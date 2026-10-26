---
title: "Air-gapped Dev Copilot"
description: "A self-hosted developer assistant for isolated environments. Level 5 Software engineering project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Air-gapped Dev Copilot"
  order: 5
---

**A self-hosted developer assistant for isolated environments.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Software engineering](/srvdb/industries/software-engineering/) | 5 of 5 | Frontier | `hnsw` |

:::note[Coming soon]
A complete, runnable **Air-gapped Dev Copilot** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Regulated teams cannot use cloud copilots.
- Code must not leave.
- Tools must be reviewable.

## How SrvDB powers Air-gapped Dev Copilot

1. SrvDB plus local models on internal hardware.
2. Indexes repositories and docs.
3. Reviewable Apache-2.0 component with SBOM.
4. Shared via thread-safe service.

## What changes

| Today | With Air-gapped Dev Copilot on SrvDB |
|---|---|
| No AI help | Local AI help |
| Code leakage risk | Zero egress |
| Opaque tools | Open source |

## How it works

1. Deploy local models.
2. Index repos.
3. Serve IDE plugin.
4. Monitor usage.
5. Update models.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.
- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Defence software
- Banking code
- Healthcare systems
- Embedded firmware
- Chip design
- Government IT

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./air-gapped-dev-copilot-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def retrieve_code(query, k=5):
    hits = db.search(embed(query), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Local model quality limits usefulness.

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
| 4 · Expert | [Coding Agent Memory](/srvdb/industries/software-engineering/coding-agent-memory/) | Long-term, crash-safe memory for autonomous coding agents. |
| 5 · Frontier | **Air-gapped Dev Copilot** (this page) | A self-hosted developer assistant for isolated environments. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
