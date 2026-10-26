---
title: "Predictive Maintenance at the Edge"
description: "Finds machines whose behaviour resembles past pre-failure signatures. Level 5 Manufacturing & IoT project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Predictive Maintenance at the Edge"
  order: 5
---

**Finds machines whose behaviour resembles past pre-failure signatures.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Manufacturing & IoT](/srvdb/industries/manufacturing-iot/) | 5 of 5 | Frontier | `sq8` |

:::note[Coming soon]
A complete, runnable **Predictive Maintenance at the Edge** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Failures are rare but costly.
- Fleet data is distributed.
- Edge hardware is limited.

## How SrvDB powers Predictive Maintenance at the Edge

1. Pre-failure signature windows from past failures are stored.
2. Live windows are matched to them on the device.
3. Fleet learning by shipping database snapshots.
4. Rust API for embedded services.

## What changes

| Today | With Predictive Maintenance at the Edge on SrvDB |
|---|---|
| Run to failure | Early warnings |
| Per-site learning | Fleet-wide signatures |
| Heavy cloud pipelines | Small edge footprint |

## How it works

1. Label past failures.
2. Extract pre-failure windows.
3. Store and ship snapshots.
4. Match live windows.
5. Schedule maintenance.

## SrvDB features it uses

- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.

## Use cases

- Bearing failures
- Pump cavitation
- Compressor faults
- Conveyor wear
- Spindle issues
- HVAC failures

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./predictive-maintenance-edge-db", dimension=128, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def pre_failure_match(window, k=5):
    hits = db.search(embed(window), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Needs enough labelled failures; validate before acting.

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
| 4 · Expert | [Plant Knowledge Agent](/srvdb/industries/manufacturing-iot/plant-knowledge-agent/) | An agent with shift memory across lines, manuals and quality records. |
| 5 · Frontier | **Predictive Maintenance at the Edge** (this page) | Finds machines whose behaviour resembles past pre-failure signatures. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
