---
title: "Ticket Deduplicator"
description: "Finds duplicate tickets and emerging issue clusters. Level 3 Customer support project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Ticket Deduplicator"
  order: 3
---

**Finds duplicate tickets and emerging issue clusters.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Customer support](/srvdb/industries/customer-support/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Ticket Deduplicator** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Outages create hundreds of duplicate tickets.
- Duplicates hide the real volume.
- New issues are spotted late.

## How SrvDB powers Ticket Deduplicator

1. New tickets are compared with open ones by cosine similarity.
2. Duplicates are linked to a parent.
3. Bursts near one vector reveal incidents.
4. `search_batch` for volume.

## What changes

| Today | With Ticket Deduplicator on SrvDB |
|---|---|
| Each duplicate handled | Handled once |
| Volume unclear | True issue counts |
| Incidents found late | Bursts detected early |

## How it works

1. Embed tickets.
2. Match against open tickets.
3. Link duplicates.
4. Alert on bursts.
5. Close in bulk.

## SrvDB features it uses

- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Outage detection
- Duplicate merging
- Knowledge-base gaps
- Bug triage
- Topic reports
- Churn-risk signals

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./ticket-deduplicator-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def duplicates_of(ticket_text, k=5):
    hits = db.search(embed(ticket_text), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Tune thresholds per product.

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
| 3 · Advanced | **Ticket Deduplicator** (this page) | Finds duplicate tickets and emerging issue clusters. |
| 4 · Expert | [Customer Memory Agent](/srvdb/industries/customer-support/customer-memory-agent/) | Remembers each customer's history across conversations. |
| 5 · Frontier | [Offline Support Kiosk](/srvdb/industries/customer-support/offline-support-kiosk/) | A self-service assistant that works without internet. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
