---
title: "Multimodal Kiosk"
description: "In-store kiosk that searches by text or photo. Level 4 Retail & e-commerce project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Multimodal Kiosk"
  order: 4
---

**In-store kiosk that searches by text or photo.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Retail & e-commerce](/srvdb/industries/retail-ecommerce/) | 4 of 5 | Expert | `hnsw` |

:::note[Coming soon]
A complete, runnable **Multimodal Kiosk** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Shoppers can't find items in large stores.
- Staff are busy.
- Store Wi-Fi is unreliable.

## How SrvDB powers Multimodal Kiosk

1. Product images and text share a multimodal embedding space in SrvDB.
2. Search by typing or taking a photo.
3. Runs offline on the kiosk.
4. Crash-safe.

## What changes

| Today | With Multimodal Kiosk on SrvDB |
|---|---|
| Asking staff | Self-service |
| Text-only search | Photo search |
| Wi-Fi dependence | Offline |

## How it works

1. Embed catalogue images and text.
2. Ship to kiosks.
3. Search locally.
4. Show aisle locations.
5. Update snapshots.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.

## Use cases

- Find by photo
- Aisle lookup
- Substitutes when out of stock
- Accessories
- Gift ideas
- Staff assistance

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./multimodal-kiosk-db", dimension=512, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def find_by_text(query, k=5):
    hits = db.search(embed(query), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Needs a multimodal embedding model.

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
| 4 · Expert | **Multimodal Kiosk** (this page) | In-store kiosk that searches by text or photo. |
| 5 · Frontier | [Edge Recommender](/srvdb/industries/retail-ecommerce/edge-recommender/) | "More like this" recommendations running on store or app hardware. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
