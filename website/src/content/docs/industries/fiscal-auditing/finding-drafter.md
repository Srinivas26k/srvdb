---
title: "Finding Drafter"
description: "Drafts audit findings and management letters from similar past findings. Level 2 Fiscal auditing project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Finding Drafter"
  order: 2
---

**Drafts audit findings and management letters from similar past findings.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Fiscal auditing](/srvdb/industries/fiscal-auditing/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Finding Drafter** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Writing findings is slow.
- Quality varies.
- Standards wording matters.

## How SrvDB powers Finding Drafter

1. Past findings and standards passages are embedded.
2. A new issue retrieves similar findings.
3. A local LLM drafts with citations.
4. Reviewers edit.

## What changes

| Today | With Finding Drafter on SrvDB |
|---|---|
| Slow drafting | Drafted findings |
| Inconsistent wording | Consistent style |
| Standards lookups | Retrieved passages |

## How it works

1. Embed findings.
2. Describe the issue.
3. Draft.
4. Review.
5. Save final.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Internal control findings
- Compliance findings
- Management letters
- Public-sector reports
- Follow-ups
- Board summaries

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./finding-drafter-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_findings(issue, k=5):
    hits = db.search(embed(issue), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Auditors own every finding.

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
| 2 · Intermediate | **Finding Drafter** (this page) | Drafts audit findings and management letters from similar past findings. |
| 3 · Advanced | [Duplicate Invoice Detector](/srvdb/industries/fiscal-auditing/duplicate-invoice-detector/) | Finds duplicate and near-duplicate invoices and vendors. |
| 4 · Expert | [Continuous Audit Agent](/srvdb/industries/fiscal-auditing/continuous-audit-agent/) | An agent monitoring entries and contracts continuously. |
| 5 · Frontier | [Public Funds Analytics](/srvdb/industries/fiscal-auditing/public-funds-analytics/) | Cross-entity pattern discovery for public-sector auditors. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
