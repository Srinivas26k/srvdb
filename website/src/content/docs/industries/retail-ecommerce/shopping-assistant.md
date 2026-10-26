---
title: "Shopping Assistant"
description: "Answers product questions from your catalogue and reviews. Level 2 Retail & e-commerce project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Shopping Assistant"
  order: 2
---

**Answers product questions from your catalogue and reviews.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Retail & e-commerce](/srvdb/industries/retail-ecommerce/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Shopping Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Customers ask questions product pages don't answer.
- Support costs rise.
- Generic bots invent facts.

## How SrvDB powers Shopping Assistant

1. Specifications, FAQs and reviews are embedded.
2. Answers cite products and reviews.
3. Local LLM keeps data in-house.
4. Updated nightly.

## What changes

| Today | With Shopping Assistant on SrvDB |
|---|---|
| Unanswered questions | Grounded answers |
| Invented facts | Cited sources |
| Support load | Self-service |

## How it works

1. Embed content.
2. Answer with citations.
3. Escalate gaps.
4. Refresh.
5. Measure.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Compatibility questions
- Size and fit
- Comparisons
- Care instructions
- Delivery questions
- Store staff help

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./shopping-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def product_answer(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Answers must cite catalogue data.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Retail & e-commerce projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Semantic Product Search](/srvdb/industries/retail-ecommerce/semantic-product-search/) | Shoppers describe what they want and find it. |
| 2 · Intermediate | **Shopping Assistant** (this page) | Answers product questions from your catalogue and reviews. |
| 3 · Advanced | [Listing Guardian](/srvdb/industries/retail-ecommerce/listing-guardian/) | Detects duplicate and counterfeit-looking listings. |
| 4 · Expert | [Multimodal Kiosk](/srvdb/industries/retail-ecommerce/multimodal-kiosk/) | In-store kiosk that searches by text or photo. |
| 5 · Frontier | [Edge Recommender](/srvdb/industries/retail-ecommerce/edge-recommender/) | "More like this" recommendations running on store or app hardware. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
