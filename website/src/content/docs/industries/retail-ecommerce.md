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
