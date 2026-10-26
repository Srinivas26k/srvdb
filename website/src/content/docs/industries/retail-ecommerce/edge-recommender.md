---
title: "Edge Recommender"
description: "\"More like this\" recommendations running on store or app hardware. Level 5 Retail & e-commerce project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Edge Recommender"
  order: 5
---

**"More like this" recommendations running on store or app hardware.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Retail & e-commerce](/srvdb/industries/retail-ecommerce/) | 5 of 5 | Frontier | `sq8` |

:::note[Coming soon]
A complete, runnable **Edge Recommender** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Recommendation services are costly.
- Cold-start items get no exposure.
- Latency matters at the point of sale.

## How SrvDB powers Edge Recommender

1. Item embeddings from content (text, image) solve cold start.
2. In-process search keeps latency low.
3. Compressed storage with `sq8`.
4. Rust or Python back-ends.

## What changes

| Today | With Edge Recommender on SrvDB |
|---|---|
| Heavy recommendation infra | A library call |
| Cold-start invisibility | Content-based similarity |
| Network latency | In-process |

## How it works

1. Embed items.
2. Store with `sq8`.
3. Recommend by item vector.
4. A/B test.
5. Refresh.

## SrvDB features it uses

- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- More like this
- Point-of-sale suggestions
- Email recommendations
- Cold-start exposure
- Trend detection
- Bundles

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./edge-recommender-db", dimension=256, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def more_like_this(item_description, k=5):
    hits = db.search(embed(item_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Benchmark latency on your hardware.

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
| 2 · Intermediate | [Shopping Assistant](/srvdb/industries/retail-ecommerce/shopping-assistant/) | Answers product questions from your catalogue and reviews. |
| 3 · Advanced | [Listing Guardian](/srvdb/industries/retail-ecommerce/listing-guardian/) | Detects duplicate and counterfeit-looking listings. |
| 4 · Expert | [Multimodal Kiosk](/srvdb/industries/retail-ecommerce/multimodal-kiosk/) | In-store kiosk that searches by text or photo. |
| 5 · Frontier | **Edge Recommender** (this page) | "More like this" recommendations running on store or app hardware. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
