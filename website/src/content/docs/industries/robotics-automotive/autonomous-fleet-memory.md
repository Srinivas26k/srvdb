---
title: "Autonomous Fleet Memory"
description: "Share learned experience across a fleet with database snapshots. Level 5 Robotics & automotive project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Autonomous Fleet Memory"
  order: 5
---

**Share learned experience across a fleet with database snapshots.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Robotics & automotive](/srvdb/industries/robotics-automotive/) | 5 of 5 | Frontier | `sq8` |

:::note[Coming soon]
A complete, runnable **Autonomous Fleet Memory** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Each vehicle learns alone.
- Connectivity is intermittent.
- Storage is limited.

## How SrvDB powers Autonomous Fleet Memory

1. Each unit keeps a local SrvDB memory.
2. Curated snapshots are shipped fleet-wide.
3. Compressed with `sq8` or `pq`.
4. Works offline in mines, at sea or in orbit.

## What changes

| Today | With Autonomous Fleet Memory on SrvDB |
|---|---|
| Isolated learning | Shared experience |
| Always-online assumption | Offline-first |
| Large storage | Compressed |

## How it works

1. Collect local memories.
2. Curate centrally.
3. Build snapshots.
4. Distribute.
5. Measure impact.

## SrvDB features it uses

- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[PQ compression (`pq`)](/srvdb/guides/index-modes/)**: stores vectors up to 32× smaller, at some cost in accuracy.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Mining fleets
- Delivery robots
- Agricultural machines
- Ships
- Drones
- Space robotics

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./autonomous-fleet-memory-db", dimension=256, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def fleet_recall(situation, k=5):
    hits = db.search(embed(situation), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Curate snapshots carefully before distribution.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Robotics & automotive projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [In-car Manual Assistant](/srvdb/industries/robotics-automotive/in-car-manual-assistant/) | Answers drivers' questions from the owner's manual, offline. |
| 2 · Intermediate | [Diagnostic Assistant](/srvdb/industries/robotics-automotive/diagnostic-assistant/) | Technicians match symptoms and trouble codes to past repairs. |
| 3 · Advanced | [Scenario Miner](/srvdb/industries/robotics-automotive/scenario-miner/) | Finds driving or robot episodes similar to a scenario of interest. |
| 4 · Expert | [Robot Episodic Memory](/srvdb/industries/robotics-automotive/robot-episodic-memory/) | Robots remember places, objects and outcomes. |
| 5 · Frontier | **Autonomous Fleet Memory** (this page) | Share learned experience across a fleet with database snapshots. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
