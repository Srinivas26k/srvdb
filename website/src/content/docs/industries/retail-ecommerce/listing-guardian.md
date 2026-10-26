---
title: "Listing Guardian"
description: "Detects duplicate and counterfeit-looking listings. Level 3 Retail & e-commerce project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Listing Guardian"
  order: 3
---

**Detects duplicate and counterfeit-looking listings.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Retail & e-commerce](/srvdb/industries/retail-ecommerce/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Listing Guardian** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Marketplaces fill with duplicate listings.
- Counterfeits copy real listings.
- Manual moderation does not scale.

## How SrvDB powers Listing Guardian

1. Listing text and images are embedded.
2. New listings are compared with existing and known-counterfeit ones.
3. High-similarity listings go to moderators.
4. Decisions fed back with `upsert`.

## What changes

| Today | With Listing Guardian on SrvDB |
|---|---|
| Manual moderation | Prioritised review |
| Duplicate clutter | Merged listings |
| Counterfeits slip through | Similarity flags |

## How it works

1. Embed listings.
2. Score new listings.
3. Queue suspects.
4. Moderate.
5. Feed back decisions.

## SrvDB features it uses

- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Duplicate listings
- Counterfeit signals
- Fake-review clusters
- Prohibited items
- Brand protection
- Catalogue hygiene

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./listing-guardian-db", dimension=512)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def suspect_listings(listing, k=5):
    hits = db.search(embed(listing), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Moderators decide.

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
| 3 · Advanced | **Listing Guardian** (this page) | Detects duplicate and counterfeit-looking listings. |
| 4 · Expert | [Multimodal Kiosk](/srvdb/industries/retail-ecommerce/multimodal-kiosk/) | In-store kiosk that searches by text or photo. |
| 5 · Frontier | [Edge Recommender](/srvdb/industries/retail-ecommerce/edge-recommender/) | "More like this" recommendations running on store or app hardware. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
