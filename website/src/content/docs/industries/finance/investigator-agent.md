---
title: "Investigator Agent"
description: "An agent that gathers context across cases, policies and customer history. Level 4 Finance & banking project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Investigator Agent"
  order: 4
---

**An agent that gathers context across cases, policies and customer history.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Finance & banking](/srvdb/industries/finance/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Investigator Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Investigators pivot across many systems.
- Notes are lost between shifts.
- Reports take hours.

## How SrvDB powers Investigator Agent

1. Separate SrvDB databases for cases, policies and customer notes.
2. The agent searches all of them per case.
3. Case memory is stored and recalled.
4. Reports are drafted from retrieved evidence.

## What changes

| Today | With Investigator Agent on SrvDB |
|---|---|
| Manual pivots | One agent gathers context |
| Lost notes | Persistent case memory |
| Slow reports | Drafted from evidence |

## How it works

1. Populate databases.
2. Agent receives a case.
3. Search all databases.
4. Draft summary with links.
5. Store memory.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Fraud investigations
- Customer complaints escalations
- Credit reviews
- Sanctions alert context
- Internal audits
- Regulatory responses

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./investigator-agent-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def case_context(case_description, k=5):
    hits = db.search(embed(case_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Humans approve every conclusion.

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
| 4 · Expert | **Investigator Agent** (this page) | An agent that gathers context across cases, policies and customer history. |
| 5 · Frontier | [On-prem Risk AI Platform](/srvdb/industries/finance/on-prem-risk-ai/) | A data-resident AI retrieval platform for risk and compliance teams. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
