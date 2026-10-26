---
title: "Calibration Drift Detector"
description: "Compares calibration snapshots stored as vectors to catch drift. Level 3 Quantum computing project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Calibration Drift Detector"
  order: 3
---

**Compares calibration snapshots stored as vectors to catch drift.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Quantum computing](/srvdb/industries/quantum-computing/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Calibration Drift Detector** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Drift is noticed late.
- Calibration history is unused.
- Devices differ.

## How SrvDB powers Calibration Drift Detector

1. Calibration parameters are stored as feature vectors per device.
2. New snapshots are compared with history.
3. Large distances flag drift.
4. One database per device.

## What changes

| Today | With Calibration Drift Detector on SrvDB |
|---|---|
| Late drift detection | Early flags |
| Unused history | Compared automatically |
| Device confusion | Per-device history |

## How it works

1. Vectorise calibration data (at least 128 values).
2. Store snapshots.
3. Compare new ones.
4. Flag drift.
5. Recalibrate.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Qubit frequency drift
- Gate fidelity changes
- Readout changes
- Noise profiles
- Device comparison
- Maintenance planning

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./calibration-drift-detector-db", dimension=128)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_calibrations(snapshot, k=5):
    hits = db.search(embed(snapshot), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Feature design determines usefulness.

:::caution
SrvDB is classical software: it stores ordinary vectors describing experiments and runs on regular computers, not on quantum hardware.
:::

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Quantum computing projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Lab Notebook Search](/srvdb/industries/quantum-computing/lab-notebook-search/) | Search the group's notes, papers and run summaries. |
| 2 · Intermediate | [Lab Q&A Assistant](/srvdb/industries/quantum-computing/lab-qa-assistant/) | Answers questions from the group's own knowledge with citations. |
| 3 · Advanced | **Calibration Drift Detector** (this page) | Compares calibration snapshots stored as vectors to catch drift. |
| 4 · Expert | [Experiment Memory Agent](/srvdb/industries/quantum-computing/experiment-memory-agent/) | An agent remembering runs, plots and decisions across experiments. |
| 5 · Frontier | [Circuit Similarity Research](/srvdb/industries/quantum-computing/circuit-similarity-research/) | Research platform for circuit embeddings and similarity search. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
