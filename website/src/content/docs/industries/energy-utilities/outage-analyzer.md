---
title: "Outage Analyzer"
description: "Finds similar past outages and their causes. Level 2 Energy & utilities project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Outage Analyzer"
  order: 2
---

**Finds similar past outages and their causes.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Energy & utilities](/srvdb/industries/energy-utilities/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Outage Analyzer** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Outages repeat in patterns.
- Reports are free text.
- Restoration needs fast context.

## How SrvDB powers Outage Analyzer

1. Outage reports are embedded.
2. New events retrieve similar past outages and fixes.
3. Clusters reveal recurring causes.
4. Updated after each event.

## What changes

| Today | With Outage Analyzer on SrvDB |
|---|---|
| Slow diagnosis | Similar past events |
| Hidden patterns | Clusters |
| Lost lessons | Recalled |

## How it works

1. Embed reports.
2. Query new events.
3. Review causes.
4. Restore.
5. Record outcomes.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Storm outages
- Equipment failures
- Vegetation faults
- Protection trips
- Substation events
- Customer impact analysis

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./outage-analyzer-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_outages(event, k=5):
    hits = db.search(embed(event), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Operators decide actions.

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
| 2 · Intermediate | **Outage Analyzer** (this page) | Finds similar past outages and their causes. |
| 3 · Advanced | [Inspection Image Matcher](/srvdb/industries/energy-utilities/inspection-image-matcher/) | Matches drone and field inspection images to known defects. |
| 4 · Expert | [Field Operations Agent](/srvdb/industries/energy-utilities/field-operations-agent/) | An offline agent for crews with procedures, asset history and notes. |
| 5 · Frontier | [Grid Analog Forecaster](/srvdb/industries/energy-utilities/grid-analog-forecaster/) | Finds historical analogues of current weather and load patterns. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
