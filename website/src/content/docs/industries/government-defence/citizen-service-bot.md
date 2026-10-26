---
title: "Citizen Service Bot"
description: "Answers citizens' questions from official public guidance. Level 2 Government & defence project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Citizen Service Bot"
  order: 2
---

**Answers citizens' questions from official public guidance.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Government & defence](/srvdb/industries/government-defence/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Citizen Service Bot** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Contact centres are overloaded.
- Guidance is spread across sites.
- Answers must be accurate and sourced.

## How SrvDB powers Citizen Service Bot

1. Public guidance pages are embedded.
2. Answers cite the official page.
3. Updates replace old guidance.
4. Runs on government infrastructure.

## What changes

| Today | With Citizen Service Bot on SrvDB |
|---|---|
| Long phone queues | Instant sourced answers |
| Outdated answers | Current guidance only |
| Scattered sites | One search |

## How it works

1. Crawl approved guidance.
2. Embed.
3. Answer with links.
4. Escalate unclear questions.
5. Refresh nightly.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Benefits questions
- Licensing
- Tax deadlines
- Permits
- Public health guidance
- Local services

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./citizen-service-bot-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def guidance_for(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Escalate anything personal or ambiguous to staff.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Government & defence projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Doctrine Assistant](/srvdb/industries/government-defence/doctrine-assistant/) | Air-gapped question answering over manuals and doctrine. |
| 2 · Intermediate | **Citizen Service Bot** (this page) | Answers citizens' questions from official public guidance. |
| 3 · Advanced | [Report Linker](/srvdb/industries/government-defence/report-linker/) | Links related reports and requests across departments. |
| 4 · Expert | [Field Intelligence Agent](/srvdb/industries/government-defence/field-intelligence-agent/) | An offline agent on rugged laptops combining text and imagery. |
| 5 · Frontier | [Sovereign AI Stack](/srvdb/industries/government-defence/sovereign-ai-stack/) | A fully reviewed, air-gapped AI stack with a published SBOM. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
