---
title: "Maintenance Assistant"
description: "Technicians describe a symptom and get the right manual section and past fixes. Level 1 Manufacturing & IoT project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Maintenance Assistant"
  order: 1
---

**Technicians describe a symptom and get the right manual section and past fixes.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Manufacturing & IoT](/srvdb/industries/manufacturing-iot/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Maintenance Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Manuals are thick and fault codes cryptic.
- Fixes live in old work orders.
- Connectivity on the floor is poor.

## How SrvDB powers Maintenance Assistant

1. Manuals and work orders are embedded in SrvDB.
2. A symptom description is the query.
3. Runs on tablets or edge boxes offline.
4. Resolved work orders are added with `upsert`.

## What changes

| Today | With Maintenance Assistant on SrvDB |
|---|---|
| Searching binders | Answers from a sentence |
| Fixes forgotten | Every past fix searchable |
| No signal on the floor | Works offline |

## How it works

1. Embed manuals and work orders.
2. Copy the database to devices.
3. Query by symptom.
4. Apply the fix.
5. Log and upsert the outcome.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Fault-code explanation
- Changeover steps
- Lubrication schedules
- Safety lockout steps
- Spare-part identification
- Shift handover

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./maintenance-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_faults(symptom, k=5):
    hits = db.search(embed(symptom), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Safety-critical steps must come from the official manual.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Manufacturing & IoT projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Maintenance Assistant** (this page) | Technicians describe a symptom and get the right manual section and past fixes. |
| 2 · Intermediate | [Defect Matcher](/srvdb/industries/manufacturing-iot/defect-matcher/) | Matches new defects to past ones, from text and images. |
| 3 · Advanced | [Sensor Anomaly Detector](/srvdb/industries/manufacturing-iot/sensor-anomaly-detector/) | Flags sensor windows far from normal operation. |
| 4 · Expert | [Plant Knowledge Agent](/srvdb/industries/manufacturing-iot/plant-knowledge-agent/) | An agent with shift memory across lines, manuals and quality records. |
| 5 · Frontier | [Predictive Maintenance at the Edge](/srvdb/industries/manufacturing-iot/predictive-maintenance-edge/) | Finds machines whose behaviour resembles past pre-failure signatures. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
