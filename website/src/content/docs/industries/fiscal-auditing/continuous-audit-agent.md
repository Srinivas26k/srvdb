---
title: "Continuous Audit Agent"
description: "An agent monitoring entries and contracts continuously. Level 4 Fiscal auditing project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Continuous Audit Agent"
  order: 4
---

**An agent monitoring entries and contracts continuously.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Fiscal auditing](/srvdb/industries/fiscal-auditing/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Continuous Audit Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Audits are periodic.
- Anomalies surface late.
- Teams are small.

## How SrvDB powers Continuous Audit Agent

1. Journal entries, contracts and findings in separate databases.
2. The agent flags unusual entries and deviations.
3. Engagement memory persists.
4. Per-entity isolation.

## What changes

| Today | With Continuous Audit Agent on SrvDB |
|---|---|
| Annual checks | Continuous checks |
| Late anomalies | Early flags |
| Small teams | Agent assistance |

## How it works

1. Connect data.
2. Index continuously.
3. Flag anomalies.
4. Review.
5. Report.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.

## Use cases

- Journal entry testing
- Contract deviations
- Expense anomalies
- Procurement signals
- Public budgets
- Grant audits

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./continuous-audit-agent-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def unusual_entries(entry_description, k=5):
    hits = db.search(embed(entry_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Auditors review every flag.

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
| 4 · Expert | **Continuous Audit Agent** (this page) | An agent monitoring entries and contracts continuously. |
| 5 · Frontier | [Public Funds Analytics](/srvdb/industries/fiscal-auditing/public-funds-analytics/) | Cross-entity pattern discovery for public-sector auditors. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
