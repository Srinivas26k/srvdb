---
title: "Helpdesk Answer Bot"
description: "Drafts replies from similar solved tickets and articles. Level 2 Customer support project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Helpdesk Answer Bot"
  order: 2
---

**Drafts replies from similar solved tickets and articles.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Customer support](/srvdb/industries/customer-support/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Helpdesk Answer Bot** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Agents rewrite the same answers daily.
- New agents lack context.
- Replies vary in quality.

## How SrvDB powers Helpdesk Answer Bot

1. Solved tickets and articles are embedded.
2. Each new ticket retrieves the closest solutions.
3. A local LLM drafts a reply citing them.
4. Agent edits are fed back.

## What changes

| Today | With Helpdesk Answer Bot on SrvDB |
|---|---|
| Writing every reply | Reviewing a draft |
| Inconsistent answers | Consistent, sourced replies |
| Slow onboarding | Context for every agent |

## How it works

1. Embed solved tickets.
2. Retrieve per ticket.
3. Draft reply.
4. Agent edits and sends.
5. Upsert the final answer.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Password and account issues
- Billing questions
- How-to questions
- Bug workarounds
- Shipping questions
- Multilingual replies

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./helpdesk-answer-bot-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def suggested_answers(ticket_text, k=5):
    hits = db.search(embed(ticket_text), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Agents approve every reply.

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
| 2 · Intermediate | **Helpdesk Answer Bot** (this page) | Drafts replies from similar solved tickets and articles. |
| 3 · Advanced | [Ticket Deduplicator](/srvdb/industries/customer-support/ticket-deduplicator/) | Finds duplicate tickets and emerging issue clusters. |
| 4 · Expert | [Customer Memory Agent](/srvdb/industries/customer-support/customer-memory-agent/) | Remembers each customer's history across conversations. |
| 5 · Frontier | [Offline Support Kiosk](/srvdb/industries/customer-support/offline-support-kiosk/) | A self-service assistant that works without internet. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
