---
title: Retail & e-commerce
description: How SrvDB powers In-store Product Finder and other local AI tools in retail & e-commerce. Use cases, architecture and code pattern.
sidebar:
  order: 9
---

Shoppers describe what they want in their own words; catalogues are organised by SKU.

:::note[Coming soon]
A complete, runnable **In-store Product Finder** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Keyword search fails on descriptive queries ("warm waterproof jacket for kids").
- "More like this" needs similarity, not categories.
- In-store kiosks and apps may run with poor connectivity.

## How SrvDB helps

- **Semantic product search** over product descriptions and images (CLIP-style embeddings).
- **Recommendations** with one `search` on an item's own vector.
- **Works offline** on kiosks and in mobile back-ends.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Product search by description
- Customer FAQ search
- Return-policy lookup
- Store inventory description search
- Catalogue lookup before adding a product
- Review search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Shopping assistant
- Grounded product comparisons
- Store staff training
- Store operations Q&A
- Seller onboarding assistant
- Size and fit advice from reviews

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Similar products
- Duplicate listings
- Counterfeit listing similarity
- Review clustering
- Fake-review signals
- Visual similarity with image embeddings

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Multimodal kiosk
- Per-shopper memory
- Catalogue sync with `upsert`
- Per-seller databases for marketplaces
- Voice shopping
- Returns triage agent

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Offline store kiosks
- Recommendations on point-of-sale hardware
- Cold-start recommendations from content embeddings
- Cross-lingual catalogues
- Trend detection: new clusters appearing in recent items
- Crash-safe memory in mobile back-ends

## Example: In-store Product Finder

A kiosk or app assistant where shoppers describe what they need and get the closest products, with "more like this" for each result.

**How it works:**

1. Embed product text (and images with a multimodal model).
2. Search with the shopper's description; show results.
3. For "more like this", search with the chosen product's vector.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

products = srvdb.Database("./products", dimension=512, mode="hnsw")

def find_products(description, k=10):
    return [i for i, _ in products.search(embed(description), k)]
```

**Recommended mode:** `hnsw` for large catalogues.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
