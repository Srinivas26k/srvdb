---
title: "Borescope Image Search"
description: "Find similar borescope images and the findings they led to. Level 4 Aerospace project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Borescope Image Search"
  order: 4
---

**Find similar borescope images and the findings they led to.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Aerospace](/srvdb/industries/aerospace/) | 4 of 5 | Expert | `hnsw` |

:::note[Coming soon]
A complete, runnable **Borescope Image Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Borescope inspections produce many images.
- Comparing to past damage is manual.
- Expertise is scarce.

## How SrvDB powers Borescope Image Search

1. Images and inspection notes embedded together.
2. New images retrieve similar past ones.
3. `hnsw` for large libraries.
4. Offline at the engine shop.

## What changes

| Today | With Borescope Image Search on SrvDB |
|---|---|
| Manual comparison | Similar images found |
| Scarce experts | Shared visual memory |
| Unused archives | Searchable images |

## How it works

1. Embed images.
2. Store with findings.
3. Query new images.
4. Inspector reviews.
5. Add outcomes.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.

## Use cases

- Blade damage
- Coating loss
- Cracks
- Burn marks
- Foreign object damage
- Training

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./borescope-image-search-db", dimension=512, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_inspections(image_description, k=5):
    hits = db.search(embed(image_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Needs a validated image embedding model; inspectors decide.

:::caution
SrvDB is not certified software. Use it for decision support and knowledge retrieval, never in flight-critical or certified systems.
:::

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Aerospace projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Maintenance Knowledge Assistant](/srvdb/industries/aerospace/maintenance-knowledge-assistant/) | Cited answers from manuals and service bulletins for technicians. |
| 2 · Intermediate | [Safety Report Analyzer](/srvdb/industries/aerospace/safety-report-analyzer/) | Clusters safety reports and links similar events. |
| 3 · Advanced | [Fleet Finding Matcher](/srvdb/industries/aerospace/fleet-finding-matcher/) | Matches new maintenance findings to similar findings across the fleet. |
| 4 · Expert | **Borescope Image Search** (this page) | Find similar borescope images and the findings they led to. |
| 5 · Frontier | [Mission Knowledge System](/srvdb/industries/aerospace/mission-knowledge-system/) | Offline mission and ground-segment knowledge for space operations. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
