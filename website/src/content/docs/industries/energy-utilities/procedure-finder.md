---
title: "Procedure Finder"
description: "Operators find the right procedure from a description. Level 1 Energy & utilities project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Procedure Finder"
  order: 1
---

**Operators find the right procedure from a description.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Energy & utilities](/srvdb/industries/energy-utilities/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Procedure Finder** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Procedures are numerous.
- Control rooms need speed.
- Field sites lack connectivity.

## How SrvDB powers Procedure Finder

1. Procedures embedded by step.
2. Descriptions retrieve procedures.
3. Offline copies for field use.
4. Updates via `upsert`.

## What changes

| Today | With Procedure Finder on SrvDB |
|---|---|
| Binder searches | Instant procedures |
| Connectivity required | Offline |
| Outdated copies | Current only |

## How it works

1. Ingest procedures.
2. Embed.
3. Search.
4. Distribute snapshots.
5. Update.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Switching procedures
- Plant start-up
- Emergency steps
- Inspection routines
- Safety rules
- Training

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./procedure-finder-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def find_procedure(description, k=5):
    hits = db.search(embed(description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Official procedures remain authoritative.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Energy & utilities projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Procedure Finder** (this page) | Operators find the right procedure from a description. |
| 2 · Intermediate | [Outage Analyzer](/srvdb/industries/energy-utilities/outage-analyzer/) | Finds similar past outages and their causes. |
| 3 · Advanced | [Inspection Image Matcher](/srvdb/industries/energy-utilities/inspection-image-matcher/) | Matches drone and field inspection images to known defects. |
| 4 · Expert | [Field Operations Agent](/srvdb/industries/energy-utilities/field-operations-agent/) | An offline agent for crews with procedures, asset history and notes. |
| 5 · Frontier | [Grid Analog Forecaster](/srvdb/industries/energy-utilities/grid-analog-forecaster/) | Finds historical analogues of current weather and load patterns. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
