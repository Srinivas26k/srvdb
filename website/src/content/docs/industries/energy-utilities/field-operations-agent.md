---
title: "Field Operations Agent"
description: "An offline agent for crews with procedures, asset history and notes. Level 4 Energy & utilities project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Field Operations Agent"
  order: 4
---

**An offline agent for crews with procedures, asset history and notes.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Energy & utilities](/srvdb/industries/energy-utilities/) | 4 of 5 | Expert | `sq8` |

:::note[Coming soon]
A complete, runnable **Field Operations Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Crews work in remote areas.
- Asset history is in the office.
- Storm response is chaotic.

## How SrvDB powers Field Operations Agent

1. Asset records, procedures and notes in separate databases on tablets.
2. The agent recalls asset history offline.
3. Notes synced later.
4. Crash-safe.

## What changes

| Today | With Field Operations Agent on SrvDB |
|---|---|
| Calling the office | Local answers |
| Paper notes | Searchable notes |
| Chaotic response | Shared knowledge |

## How it works

1. Preload databases.
2. Use offline.
3. Capture notes.
4. Sync.
5. Review.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.

## Use cases

- Storm response
- Asset maintenance
- Meter issues
- Safety checks
- New-crew support
- Substation visits

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./field-operations-agent-db", dimension=384, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def asset_recall(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Safety procedures follow official rules.

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
| 4 · Expert | **Field Operations Agent** (this page) | An offline agent for crews with procedures, asset history and notes. |
| 5 · Frontier | [Grid Analog Forecaster](/srvdb/industries/energy-utilities/grid-analog-forecaster/) | Finds historical analogues of current weather and load patterns. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
