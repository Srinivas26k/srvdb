---
title: Government & defence
description: How SrvDB powers Air-gapped Knowledge Assistant and other local AI tools in government & defence. Use cases, architecture and code pattern.
sidebar:
  order: 5
---

Classified and isolated networks cannot use cloud AI at all.

:::note[Coming soon]
A complete, runnable **Air-gapped Knowledge Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Doctrine, manuals and reports are scattered and hard to search.
- Systems must run with no internet access, ever.
- Every dependency has to be reviewed.

## How SrvDB helps

- **Fully offline:** no network access, no telemetry, no background service.
- **Small, reviewable footprint:** one library, no Python runtime dependencies, Apache-2.0, a published SBOM with every release.
- **Crash-safe:** committed data survives power loss on field hardware.

## Example: Air-gapped Knowledge Assistant

A question-answering assistant over manuals and reports that runs on a single isolated machine, with every answer linked to its source passages.

**How it works:**

1. Install from a wheel copied into the isolated network.
2. Embed documents with a local model; store source and section as metadata.
3. Answer questions with a local LLM, always showing sources.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

library = srvdb.Database("./library", dimension=768)

def cite(question, k=5):
    return [json.loads(library.get(i))["source"] for i, _ in library.search(embed(question), k)]
```

**Recommended mode:** `flat` or `hnsw` depending on size.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
