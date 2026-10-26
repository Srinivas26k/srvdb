---
title: "Grid Analog Forecaster"
description: "Finds historical analogues of current weather and load patterns. Level 5 Energy & utilities project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Grid Analog Forecaster"
  order: 5
---

**Finds historical analogues of current weather and load patterns.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Energy & utilities](/srvdb/industries/energy-utilities/) | 5 of 5 | Frontier | `flat` |

:::note[Coming soon]
A complete, runnable **Grid Analog Forecaster** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Forecasting extremes is hard.
- History is underused.
- Renewables add variability.

## How SrvDB powers Grid Analog Forecaster

1. Weather and load patterns are stored as feature vectors.
2. Current conditions retrieve the nearest historical analogues.
3. Planners review what happened then.
4. Air-gapped on OT networks.

## What changes

| Today | With Grid Analog Forecaster on SrvDB |
|---|---|
| Model-only forecasts | Plus historical analogues |
| Unused history | Searchable history |
| Network exposure | Air-gapped |

## How it works

1. Vectorise historical patterns (at least 128 values).
2. Store.
3. Query current conditions.
4. Review analogues.
5. Plan.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.

## Use cases

- Heatwave load
- Storm preparation
- Wind ramps
- Solar variability
- Hydro planning
- Demand response

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./grid-analog-forecaster-db", dimension=128)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def historical_analogues(conditions, k=5):
    hits = db.search(embed(conditions), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Analogues complement, not replace, forecasting models.

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
| 3 · Advanced | [Inspection Image Matcher](/srvdb/industries/energy-utilities/inspection-image-matcher/) | Matches drone and field inspection images to known defects. |
| 4 · Expert | [Field Operations Agent](/srvdb/industries/energy-utilities/field-operations-agent/) | An offline agent for crews with procedures, asset history and notes. |
| 5 · Frontier | **Grid Analog Forecaster** (this page) | Finds historical analogues of current weather and load patterns. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
