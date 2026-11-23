---
title: "Fraud Pattern Matcher"
description: "Compares new cases with confirmed fraud cases and ranks them for investigators. Level 3 Finance & banking project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Fraud Pattern Matcher"
  order: 3
---

**Compares new cases with confirmed fraud cases and ranks them for investigators.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Finance & banking](/srvdb/industries/finance/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Fraud Pattern Matcher** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- New fraud resembles old fraud, but not exactly.
- Investigators are overloaded.
- Rules lag behind schemes.

## How SrvDB powers Fraud Pattern Matcher

1. Confirmed case narratives are embedded with outcomes.
2. New cases are scored against them with cosine similarity.
3. High-similarity cases go to investigators first.
4. Outcomes feed back with `upsert`.

## What changes

| Today | With Fraud Pattern Matcher on SrvDB |
|---|---|
| FIFO case queues | Risk-ranked queues |
| Static rules | Learns from outcomes |
| Isolated cases | Links to similar past cases |

## How it works

1. Embed confirmed cases.
2. Score new cases.
3. Rank queue.
4. Investigate.
5. Feed back outcomes.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Card fraud narratives
- Account takeover patterns
- Loan application fraud
- Insurance claim fraud
- Mule-account behaviour
- Scam reports

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./fraud-pattern-matcher-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def resembles_fraud(case_narrative, k=5):
    hits = db.search(embed(case_narrative), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Decision support; investigators decide.

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
| 3 · Advanced | **Fraud Pattern Matcher** (this page) | Compares new cases with confirmed fraud cases and ranks them for investigators. |
| 4 · Expert | [Investigator Agent](/srvdb/industries/finance/investigator-agent/) | An agent that gathers context across cases, policies and customer history. |
| 5 · Frontier | [On-prem Risk AI Platform](/srvdb/industries/finance/on-prem-risk-ai/) | A data-resident AI retrieval platform for risk and compliance teams. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
