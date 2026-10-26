---
title: "Semantic Product Search"
description: "Shoppers describe what they want and find it. Level 1 Retail & e-commerce project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Semantic Product Search"
  order: 1
---

**Shoppers describe what they want and find it.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Retail & e-commerce](/srvdb/industries/retail-ecommerce/) | 1 of 5 | Beginner | `hnsw` |

:::note[Coming soon]
A complete, runnable **Semantic Product Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Keyword search fails on descriptive queries.
- Zero-result searches lose sales.
- Catalogues use internal jargon.

## How SrvDB powers Semantic Product Search

1. Product descriptions are embedded in SrvDB.
2. Queries in natural language find the closest products.
3. `hnsw` for large catalogues.
4. Catalogue updates via `upsert`.

## What changes

| Today | With Semantic Product Search on SrvDB |
|---|---|
| Zero results | Relevant results |
| Jargon-only search | Customers' own words |
| Manual synonyms | Meaning-based matching |

## How it works

1. Embed products.
2. Serve search.
3. Track queries.
4. Update catalogue.
5. Measure conversion.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Descriptive search
- Gift finder
- Long-tail queries
- Multilingual search
- B2B part search
- Internal catalogue search

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./semantic-product-search-db", dimension=384, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def find_products(description, k=5):
    hits = db.search(embed(description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Check `hnsw` recall against `flat`.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Retail & e-commerce projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Semantic Product Search** (this page) | Shoppers describe what they want and find it. |
| 2 · Intermediate | [Shopping Assistant](/srvdb/industries/retail-ecommerce/shopping-assistant/) | Answers product questions from your catalogue and reviews. |
| 3 · Advanced | [Listing Guardian](/srvdb/industries/retail-ecommerce/listing-guardian/) | Detects duplicate and counterfeit-looking listings. |
| 4 · Expert | [Multimodal Kiosk](/srvdb/industries/retail-ecommerce/multimodal-kiosk/) | In-store kiosk that searches by text or photo. |
| 5 · Frontier | [Edge Recommender](/srvdb/industries/retail-ecommerce/edge-recommender/) | "More like this" recommendations running on store or app hardware. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
