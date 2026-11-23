---
title: "Robot Episodic Memory"
description: "Robots remember places, objects and outcomes. Level 4 Robotics & automotive project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Robot Episodic Memory"
  order: 4
---

**Robots remember places, objects and outcomes.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Robotics & automotive](/srvdb/industries/robotics-automotive/) | 4 of 5 | Expert | `sq8` |

:::note[Coming soon]
A complete, runnable **Robot Episodic Memory** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Robots forget past experience.
- Cloud memory adds latency.
- Power loss wipes state.

## How SrvDB powers Robot Episodic Memory

1. Observations are embedded and stored on the robot.
2. Recall by current context.
3. Crash-safe `persist()`.
4. Rust API in the robot stack.

## What changes

| Today | With Robot Episodic Memory on SrvDB |
|---|---|
| Stateless robots | Experience memory |
| Cloud latency | In-process recall |
| State lost on power cut | Crash-safe |

## How it works

1. Embed observations.
2. Store locally.
3. Recall in planning.
4. Persist at safe points.
5. Compact periodically.

## SrvDB features it uses

- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.

## Use cases

- Place recognition
- Object memory
- Task outcomes
- Human preferences
- Navigation hints
- Failure avoidance

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./robot-episodic-memory-db", dimension=256, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def recall(observation, k=5):
    hits = db.search(embed(observation), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Validate memory use in safety-relevant behaviour.

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
| 4 · Expert | **Robot Episodic Memory** (this page) | Robots remember places, objects and outcomes. |
| 5 · Frontier | [Autonomous Fleet Memory](/srvdb/industries/robotics-automotive/autonomous-fleet-memory/) | Share learned experience across a fleet with database snapshots. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
