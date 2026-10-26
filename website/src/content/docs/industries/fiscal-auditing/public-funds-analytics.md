---
title: "Public Funds Analytics"
description: "Cross-entity pattern discovery for public-sector auditors. Level 5 Fiscal auditing project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Public Funds Analytics"
  order: 5
---

**Cross-entity pattern discovery for public-sector auditors.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Fiscal auditing](/srvdb/industries/fiscal-auditing/) | 5 of 5 | Frontier | `hnsw` |

:::note[Coming soon]
A complete, runnable **Public Funds Analytics** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Public spending spans many entities.
- Patterns cross organisational boundaries.
- Archives span years.

## How SrvDB powers Public Funds Analytics

1. Multi-year records from many entities are embedded.
2. Similar patterns are linked across entities.
3. Offline field audits supported.
4. Crash-safe evidence index.

## What changes

| Today | With Public Funds Analytics on SrvDB |
|---|---|
| Entity silos | Cross-entity patterns |
| Year-by-year view | Multi-year search |
| Office-only tools | Field-capable |

## How it works

1. Ingest records.
2. Embed.
3. Link patterns.
4. Review.
5. Report.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.

## Use cases

- Procurement patterns
- Grant misuse signals
- Payroll anomalies
- Asset audits
- Performance audits
- Multilingual audits

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./public-funds-analytics-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def related_patterns(record, k=5):
    hits = db.search(embed(record), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Signals require auditor judgement.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Fiscal auditing projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Working-paper Search](/srvdb/industries/fiscal-auditing/working-paper-search/) | Search prior working papers and findings by meaning. |
| 2 · Intermediate | [Finding Drafter](/srvdb/industries/fiscal-auditing/finding-drafter/) | Drafts audit findings and management letters from similar past findings. |
| 3 · Advanced | [Duplicate Invoice Detector](/srvdb/industries/fiscal-auditing/duplicate-invoice-detector/) | Finds duplicate and near-duplicate invoices and vendors. |
| 4 · Expert | [Continuous Audit Agent](/srvdb/industries/fiscal-auditing/continuous-audit-agent/) | An agent monitoring entries and contracts continuously. |
| 5 · Frontier | **Public Funds Analytics** (this page) | Cross-entity pattern discovery for public-sector auditors. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
