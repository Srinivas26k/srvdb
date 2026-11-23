---
title: "Sensor Anomaly Detector"
description: "Flags sensor windows far from normal operation. Level 3 Manufacturing & IoT project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Sensor Anomaly Detector"
  order: 3
---

**Flags sensor windows far from normal operation.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Manufacturing & IoT](/srvdb/industries/manufacturing-iot/) | 3 of 5 | Advanced | `sq8` |

:::note[Coming soon]
A complete, runnable **Sensor Anomaly Detector** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Thresholds miss subtle patterns.
- Failures are expensive.
- Edge devices are small.

## How SrvDB powers Sensor Anomaly Detector

1. Normal sensor windows are stored as feature vectors with `sq8`.
2. New windows are compared; a large distance is an anomaly.
3. Runs on ARM edge devices.
4. Crash-safe through power cuts.

## What changes

| Today | With Sensor Anomaly Detector on SrvDB |
|---|---|
| Static thresholds | Learned normal behaviour |
| Late failure detection | Early deviations |
| Cloud dependence | On the device |

## How it works

1. Collect normal windows.
2. Store in SrvDB (`sq8`).
3. Score live windows.
4. Alert operators.
5. Refresh baseline.

## SrvDB features it uses

- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.

## Use cases

- Vibration anomalies
- Temperature drift
- Pressure patterns
- Motor current signatures
- Energy anomalies
- Process deviations

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./sensor-anomaly-detector-db", dimension=128, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def deviation(window, k=5):
    hits = db.search(embed(window), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Requires a representative baseline and tuning.

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
| 3 · Advanced | **Sensor Anomaly Detector** (this page) | Flags sensor windows far from normal operation. |
| 4 · Expert | [Plant Knowledge Agent](/srvdb/industries/manufacturing-iot/plant-knowledge-agent/) | An agent with shift memory across lines, manuals and quality records. |
| 5 · Frontier | [Predictive Maintenance at the Edge](/srvdb/industries/manufacturing-iot/predictive-maintenance-edge/) | Finds machines whose behaviour resembles past pre-failure signatures. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
