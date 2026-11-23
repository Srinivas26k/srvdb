---
title: "Complaint Analyzer"
description: "Clusters and deduplicates customer complaints to find root causes. Level 2 Finance & banking project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Complaint Analyzer"
  order: 2
---

**Clusters and deduplicates customer complaints to find root causes.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Finance & banking](/srvdb/industries/finance/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Complaint Analyzer** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Complaints arrive in many channels and words.
- Root causes hide in volume.
- Regulators expect trend analysis.

## How SrvDB powers Complaint Analyzer

1. Complaints are embedded and grouped by nearest neighbours.
2. Duplicates across channels are linked.
3. New complaints far from known groups are flagged.
4. `search_batch` processes daily volumes.

## What changes

| Today | With Complaint Analyzer on SrvDB |
|---|---|
| Manual tagging | Automatic grouping |
| Duplicates double-counted | Linked across channels |
| Trends spotted late | New clusters surface quickly |

## How it works

1. Ingest complaints.
2. Embed.
3. Group and deduplicate.
4. Report trends.
5. Track remediation.

## SrvDB features it uses

- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Root-cause analysis
- Regulatory reporting
- Product issue detection
- Channel deduplication
- Vulnerable-customer signals
- Remediation tracking

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./complaint-analyzer-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_complaints(complaint, k=5):
    hits = db.search(embed(complaint), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Human review of groupings.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Finance & banking projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Policy Assistant](/srvdb/industries/finance/policy-assistant/) | Answers staff questions from internal policies and product terms. |
| 2 · Intermediate | **Complaint Analyzer** (this page) | Clusters and deduplicates customer complaints to find root causes. |
| 3 · Advanced | [Fraud Pattern Matcher](/srvdb/industries/finance/fraud-pattern-matcher/) | Compares new cases with confirmed fraud cases and ranks them for investigators. |
| 4 · Expert | [Investigator Agent](/srvdb/industries/finance/investigator-agent/) | An agent that gathers context across cases, policies and customer history. |
| 5 · Frontier | [On-prem Risk AI Platform](/srvdb/industries/finance/on-prem-risk-ai/) | A data-resident AI retrieval platform for risk and compliance teams. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
