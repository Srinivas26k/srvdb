---
title: "Offline Support Kiosk"
description: "A self-service assistant that works without internet. Level 5 Customer support project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Offline Support Kiosk"
  order: 5
---

**A self-service assistant that works without internet.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Customer support](/srvdb/industries/customer-support/) | 5 of 5 | Frontier | `flat` |

:::note[Coming soon]
A complete, runnable **Offline Support Kiosk** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Stores, factories and remote sites have poor connectivity.
- Staff need answers anyway.
- Devices lose power.

## How SrvDB powers Offline Support Kiosk

1. The knowledge base is shipped as a SrvDB folder.
2. Search runs locally.
3. Crash-safe storage survives power loss.
4. Updates are new folder snapshots.

## What changes

| Today | With Offline Support Kiosk on SrvDB |
|---|---|
| No help offline | Full help offline |
| Paper manuals | Searchable knowledge |
| Corrupted local data | Crash-safe |

## How it works

1. Build the knowledge base centrally.
2. Ship snapshots.
3. Search locally.
4. Collect questions offline.
5. Sync later.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.

## Use cases

- Retail stores
- Factory floors
- Ships
- Rural service centres
- Appliances
- Events

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./offline-support-kiosk-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def kiosk_answer(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Snapshots must be refreshed regularly.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Customer support projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Knowledge Base Search](/srvdb/industries/customer-support/kb-search/) | Customers and agents find the right article from their own words. |
| 2 · Intermediate | [Helpdesk Answer Bot](/srvdb/industries/customer-support/helpdesk-answer-bot/) | Drafts replies from similar solved tickets and articles. |
| 3 · Advanced | [Ticket Deduplicator](/srvdb/industries/customer-support/ticket-deduplicator/) | Finds duplicate tickets and emerging issue clusters. |
| 4 · Expert | [Customer Memory Agent](/srvdb/industries/customer-support/customer-memory-agent/) | Remembers each customer's history across conversations. |
| 5 · Frontier | **Offline Support Kiosk** (this page) | A self-service assistant that works without internet. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
