---
title: "Inspection Image Matcher"
description: "Matches drone and field inspection images to known defects. Level 3 Energy & utilities project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Inspection Image Matcher"
  order: 3
---

**Matches drone and field inspection images to known defects.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Energy & utilities](/srvdb/industries/energy-utilities/) | 3 of 5 | Advanced | `hnsw` |

:::note[Coming soon]
A complete, runnable **Inspection Image Matcher** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Drone inspections produce huge image sets.
- Defects are subtle.
- Review is slow.

## How SrvDB powers Inspection Image Matcher

1. Images embedded with defect labels.
2. New images retrieve similar defects.
3. `hnsw` for scale; `sq8` for size.
4. Reviewed by engineers.

## What changes

| Today | With Inspection Image Matcher on SrvDB |
|---|---|
| Manual image review | Prioritised images |
| Subtle defects missed | Similar defects surfaced |
| Huge archives | Compact vectors |

## How it works

1. Embed labelled images.
2. Embed new images.
3. Match.
4. Review.
5. Add labels.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Insulator damage
- Corrosion
- Vegetation encroachment
- Solar panel defects
- Wind blade damage
- Pipeline inspection

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./inspection-image-matcher-db", dimension=512, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_defects(image_description, k=5):
    hits = db.search(embed(image_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Needs a validated image embedding model.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Energy & utilities projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Procedure Finder](/srvdb/industries/energy-utilities/procedure-finder/) | Operators find the right procedure from a description. |
| 2 · Intermediate | [Outage Analyzer](/srvdb/industries/energy-utilities/outage-analyzer/) | Finds similar past outages and their causes. |
| 3 · Advanced | **Inspection Image Matcher** (this page) | Matches drone and field inspection images to known defects. |
| 4 · Expert | [Field Operations Agent](/srvdb/industries/energy-utilities/field-operations-agent/) | An offline agent for crews with procedures, asset history and notes. |
| 5 · Frontier | [Grid Analog Forecaster](/srvdb/industries/energy-utilities/grid-analog-forecaster/) | Finds historical analogues of current weather and load patterns. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
