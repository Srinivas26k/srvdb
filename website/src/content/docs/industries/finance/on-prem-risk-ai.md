---
title: "On-prem Risk AI Platform"
description: "A data-resident AI retrieval platform for risk and compliance teams. Level 5 Finance & banking project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 On-prem Risk AI Platform"
  order: 5
---

**A data-resident AI retrieval platform for risk and compliance teams.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Finance & banking](/srvdb/industries/finance/) | 5 of 5 | Frontier | `hnsw` |

:::note[Coming soon]
A complete, runnable **On-prem Risk AI Platform** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Data residency limits cloud AI.
- Many teams, duplicated efforts.
- Model-risk teams need traceability.

## How SrvDB powers On-prem Risk AI Platform

1. SrvDB databases per team on bank infrastructure.
2. Common retrieval API with audit logs.
3. Crash-safe storage and versioned on-disk formats.
4. Thread-safe for concurrent users.

## What changes

| Today | With On-prem Risk AI Platform on SrvDB |
|---|---|
| Shadow AI tools | One governed platform |
| No traceability | Logged retrievals |
| Residency issues | Data stays in-country |

## How it works

1. Deploy local models.
2. Provision databases.
3. Expose API.
4. Log retrievals.
5. Review with model-risk teams.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.

## Use cases

- Model documentation retrieval
- Stress-scenario analogues
- Credit memo drafting
- Policy Q&A
- Trade-surveillance review
- Audit evidence search

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./on-prem-risk-ai-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def retrieve(query, k=5):
    hits = db.search(embed(query), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Benchmark latency before using in transaction flows.

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
| 2 · Intermediate | [Complaint Analyzer](/srvdb/industries/finance/complaint-analyzer/) | Clusters and deduplicates customer complaints to find root causes. |
| 3 · Advanced | [Fraud Pattern Matcher](/srvdb/industries/finance/fraud-pattern-matcher/) | Compares new cases with confirmed fraud cases and ranks them for investigators. |
| 4 · Expert | [Investigator Agent](/srvdb/industries/finance/investigator-agent/) | An agent that gathers context across cases, policies and customer history. |
| 5 · Frontier | **On-prem Risk AI Platform** (this page) | A data-resident AI retrieval platform for risk and compliance teams. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
