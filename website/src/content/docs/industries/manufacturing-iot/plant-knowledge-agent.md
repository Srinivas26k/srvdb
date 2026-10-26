---
title: "Plant Knowledge Agent"
description: "An agent with shift memory across lines, manuals and quality records. Level 4 Manufacturing & IoT project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Plant Knowledge Agent"
  order: 4
---

**An agent with shift memory across lines, manuals and quality records.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Manufacturing & IoT](/srvdb/industries/manufacturing-iot/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Plant Knowledge Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Knowledge is split per line and shift.
- Handover loses context.
- Digital twins lack a knowledge layer.

## How SrvDB powers Plant Knowledge Agent

1. One SrvDB database per line plus shared manuals.
2. The agent remembers shift events.
3. Answers combine manuals, events and quality records.
4. Database snapshots ship to edge gateways.

## What changes

| Today | With Plant Knowledge Agent on SrvDB |
|---|---|
| Lost handover context | Persistent shift memory |
| Line silos | Plant-wide knowledge |
| Static twins | A searchable knowledge layer |

## How it works

1. Create databases per line.
2. Log events.
3. Agent queries all.
4. Draft handovers.
5. Ship snapshots.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Shift handover
- Downtime analysis
- Quality investigations
- New-starter training
- Digital-twin queries
- Continuous improvement

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./plant-knowledge-agent-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def plant_recall(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Agent output reviewed by supervisors.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Manufacturing & IoT projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Maintenance Assistant](/srvdb/industries/manufacturing-iot/maintenance-assistant/) | Technicians describe a symptom and get the right manual section and past fixes. |
| 2 · Intermediate | [Defect Matcher](/srvdb/industries/manufacturing-iot/defect-matcher/) | Matches new defects to past ones, from text and images. |
| 3 · Advanced | [Sensor Anomaly Detector](/srvdb/industries/manufacturing-iot/sensor-anomaly-detector/) | Flags sensor windows far from normal operation. |
| 4 · Expert | **Plant Knowledge Agent** (this page) | An agent with shift memory across lines, manuals and quality records. |
| 5 · Frontier | [Predictive Maintenance at the Edge](/srvdb/industries/manufacturing-iot/predictive-maintenance-edge/) | Finds machines whose behaviour resembles past pre-failure signatures. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
