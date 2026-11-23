---
title: "Secure Leak Analysis"
description: "Air-gapped analysis of large leaked datasets. Level 5 Media & journalism project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Secure Leak Analysis"
  order: 5
---

**Air-gapped analysis of large leaked datasets.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Media & journalism](/srvdb/industries/media-journalism/) | 5 of 5 | Frontier | `hnsw` |

:::note[Coming soon]
A complete, runnable **Secure Leak Analysis** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Leaks can be huge and dangerous to handle online.
- Many languages.
- Verification is critical.

## How SrvDB powers Secure Leak Analysis

1. Fully offline SrvDB with local multilingual embeddings.
2. Cross-lingual search links documents.
3. Crash-safe on field laptops.
4. Logged retrievals.

## What changes

| Today | With Secure Leak Analysis on SrvDB |
|---|---|
| Online tools risk exposure | Offline only |
| Language barriers | Cross-lingual search |
| Fragile setups | Crash-safe |

## How it works

1. Set up an isolated machine.
2. Ingest.
3. Search across languages.
4. Verify.
5. Publish carefully.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.

## Use cases

- Cross-border leaks
- Financial records
- Corporate documents
- Government files
- Image archives
- Collaborative investigations

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./secure-leak-analysis-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def leak_search(query, k=5):
    hits = db.search(embed(query), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Legal and ethical review is essential.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Media & journalism projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Archive Search](/srvdb/industries/media-journalism/archive-search/) | Search decades of articles by meaning. |
| 2 · Intermediate | [Fact-check Assistant](/srvdb/industries/media-journalism/fact-check-assistant/) | Matches claims against previously fact-checked claims and sources. |
| 3 · Advanced | [Narrative Tracker](/srvdb/industries/media-journalism/narrative-tracker/) | Clusters stories and posts into narratives. |
| 4 · Expert | [Investigation Workbench](/srvdb/industries/media-journalism/investigation-workbench/) | Isolated, multimodal workspace per investigation. |
| 5 · Frontier | **Secure Leak Analysis** (this page) | Air-gapped analysis of large leaked datasets. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
