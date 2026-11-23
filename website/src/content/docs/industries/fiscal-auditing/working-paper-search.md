---
title: "Working-paper Search"
description: "Search prior working papers and findings by meaning. Level 1 Fiscal auditing project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Working-paper Search"
  order: 1
---

**Search prior working papers and findings by meaning.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Fiscal auditing](/srvdb/industries/fiscal-auditing/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Working-paper Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Prior-year papers are hard to reuse.
- Findings repeat across entities.
- Keyword search fails.

## How SrvDB powers Working-paper Search

1. Working papers and findings are embedded.
2. Audit areas query similar past work.
3. Per-engagement databases.
4. Local storage.

## What changes

| Today | With Working-paper Search on SrvDB |
|---|---|
| Starting from scratch | Reusing prior work |
| Repeated findings missed | Patterns visible |
| Keyword search | Meaning search |

## How it works

1. Ingest papers.
2. Embed.
3. Search by area.
4. Reuse.
5. Archive.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Planning
- Risk assessment
- Testing approaches
- Prior findings
- Standards references
- Training

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./working-paper-search-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def prior_work(audit_area, k=5):
    hits = db.search(embed(audit_area), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Independence rules apply to sharing.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Fiscal auditing projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Working-paper Search** (this page) | Search prior working papers and findings by meaning. |
| 2 · Intermediate | [Finding Drafter](/srvdb/industries/fiscal-auditing/finding-drafter/) | Drafts audit findings and management letters from similar past findings. |
| 3 · Advanced | [Duplicate Invoice Detector](/srvdb/industries/fiscal-auditing/duplicate-invoice-detector/) | Finds duplicate and near-duplicate invoices and vendors. |
| 4 · Expert | [Continuous Audit Agent](/srvdb/industries/fiscal-auditing/continuous-audit-agent/) | An agent monitoring entries and contracts continuously. |
| 5 · Frontier | [Public Funds Analytics](/srvdb/industries/fiscal-auditing/public-funds-analytics/) | Cross-entity pattern discovery for public-sector auditors. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
