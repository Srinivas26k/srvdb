---
title: "Duplicate Invoice Detector"
description: "Finds duplicate and near-duplicate invoices and vendors. Level 3 Fiscal auditing project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Duplicate Invoice Detector"
  order: 3
---

**Finds duplicate and near-duplicate invoices and vendors.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Fiscal auditing](/srvdb/industries/fiscal-auditing/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Duplicate Invoice Detector** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Duplicates hide behind different wording.
- Vendor masters have duplicates.
- Manual checks are sampled.

## How SrvDB powers Duplicate Invoice Detector

1. Invoice and vendor descriptions are embedded.
2. Near-duplicates exceed a cosine threshold.
3. Full-population checks with `search_batch`.
4. Results reviewed.

## What changes

| Today | With Duplicate Invoice Detector on SrvDB |
|---|---|
| Sampling | Full population |
| Exact-match checks | Near-duplicate checks |
| Manual review | Prioritised review |

## How it works

1. Embed invoices.
2. Check each against others.
3. Flag above threshold.
4. Review.
5. Recover.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Duplicate payments
- Vendor duplicates
- Split invoices
- Expense claims
- Contract overlaps
- Recovery audits

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./duplicate-invoice-detector-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def duplicate_candidates(description, k=5):
    hits = db.search(embed(description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Amounts and dates still need rule checks.

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
| 3 · Advanced | **Duplicate Invoice Detector** (this page) | Finds duplicate and near-duplicate invoices and vendors. |
| 4 · Expert | [Continuous Audit Agent](/srvdb/industries/fiscal-auditing/continuous-audit-agent/) | An agent monitoring entries and contracts continuously. |
| 5 · Frontier | [Public Funds Analytics](/srvdb/industries/fiscal-auditing/public-funds-analytics/) | Cross-entity pattern discovery for public-sector auditors. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
