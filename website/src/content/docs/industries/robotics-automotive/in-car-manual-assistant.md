---
title: "In-car Manual Assistant"
description: "Answers drivers' questions from the owner's manual, offline. Level 1 Robotics & automotive project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 In-car Manual Assistant"
  order: 1
---

**Answers drivers' questions from the owner's manual, offline.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Robotics & automotive](/srvdb/industries/robotics-automotive/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **In-car Manual Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Manuals are long and rarely read.
- Cars are often offline.
- Answers must be accurate.

## How SrvDB powers In-car Manual Assistant

1. The manual is embedded and stored in the vehicle.
2. Spoken questions become searches.
3. Answers cite manual sections.
4. Runs without connectivity.

## What changes

| Today | With In-car Manual Assistant on SrvDB |
|---|---|
| Glovebox manuals | Spoken answers |
| Cloud dependence | Offline |
| Guesswork | Cited sections |

## How it works

1. Embed manual.
2. Ship with firmware.
3. Answer questions.
4. Log unanswered ones.
5. Update with releases.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Warning-light explanations
- Feature how-tos
- Maintenance intervals
- Tyre pressure
- Charging (EVs)
- Infotainment help

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./in-car-manual-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def manual_answer(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Safety-critical instructions follow official procedures.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Robotics & automotive projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **In-car Manual Assistant** (this page) | Answers drivers' questions from the owner's manual, offline. |
| 2 · Intermediate | [Diagnostic Assistant](/srvdb/industries/robotics-automotive/diagnostic-assistant/) | Technicians match symptoms and trouble codes to past repairs. |
| 3 · Advanced | [Scenario Miner](/srvdb/industries/robotics-automotive/scenario-miner/) | Finds driving or robot episodes similar to a scenario of interest. |
| 4 · Expert | [Robot Episodic Memory](/srvdb/industries/robotics-automotive/robot-episodic-memory/) | Robots remember places, objects and outcomes. |
| 5 · Frontier | [Autonomous Fleet Memory](/srvdb/industries/robotics-automotive/autonomous-fleet-memory/) | Share learned experience across a fleet with database snapshots. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
