---
title: "Customer Memory Agent"
description: "Remembers each customer's history across conversations. Level 4 Customer support project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Customer Memory Agent"
  order: 4
---

**Remembers each customer's history across conversations.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Customer support](/srvdb/industries/customer-support/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Customer Memory Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Customers repeat themselves to every agent.
- History is spread across systems.
- Context is lost between channels.

## How SrvDB powers Customer Memory Agent

1. One SrvDB database per tenant, or per customer segment.
2. Conversation summaries are stored and recalled by meaning.
3. Screenshots matched with image embeddings.
4. Old data removed with `delete` and `compact()`.

## What changes

| Today | With Customer Memory Agent on SrvDB |
|---|---|
| Customers repeat themselves | Context recalled automatically |
| Channel silos | One memory |
| Retention handled manually | Delete and compact by policy |

## How it works

1. Summarise conversations.
2. Store per customer.
3. Recall on contact.
4. Update after each case.
5. Apply retention.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Personalised support
- Escalation context
- Account managers' briefings
- SaaS tenant isolation
- Voice transcript recall
- Retention compliance

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./customer-memory-agent-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def recall_customer(context, k=5):
    hits = db.search(embed(context), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Respect privacy and retention rules.

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
| 4 · Expert | **Customer Memory Agent** (this page) | Remembers each customer's history across conversations. |
| 5 · Frontier | [Offline Support Kiosk](/srvdb/industries/customer-support/offline-support-kiosk/) | A self-service assistant that works without internet. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
