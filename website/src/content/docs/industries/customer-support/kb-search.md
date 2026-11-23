---
title: "Knowledge Base Search"
description: "Customers and agents find the right article from their own words. Level 1 Customer support project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Knowledge Base Search"
  order: 1
---

**Customers and agents find the right article from their own words.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Customer support](/srvdb/industries/customer-support/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Knowledge Base Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Customers describe problems differently from article titles.
- Agents search several tools.
- Good articles go unused.

## How SrvDB powers Knowledge Base Search

1. Every article section is embedded in SrvDB.
2. Any question finds the closest sections.
3. Article updates use `upsert`.
4. Runs inside your help centre backend.

## What changes

| Today | With Knowledge Base Search on SrvDB |
|---|---|
| Keyword search fails on real questions | Search by meaning |
| Agents switch tools | One search box |
| Articles unused | Surfaced when relevant |

## How it works

1. Export articles.
2. Embed sections.
3. Serve search.
4. Track unanswered queries.
5. Update articles.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Customer self-service
- Agent lookup
- In-app help
- Release-note search
- Community answers
- Internal IT help

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./kb-search-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def find_articles(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Article quality limits answer quality.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Customer support projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Knowledge Base Search** (this page) | Customers and agents find the right article from their own words. |
| 2 · Intermediate | [Helpdesk Answer Bot](/srvdb/industries/customer-support/helpdesk-answer-bot/) | Drafts replies from similar solved tickets and articles. |
| 3 · Advanced | [Ticket Deduplicator](/srvdb/industries/customer-support/ticket-deduplicator/) | Finds duplicate tickets and emerging issue clusters. |
| 4 · Expert | [Customer Memory Agent](/srvdb/industries/customer-support/customer-memory-agent/) | Remembers each customer's history across conversations. |
| 5 · Frontier | [Offline Support Kiosk](/srvdb/industries/customer-support/offline-support-kiosk/) | A self-service assistant that works without internet. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
