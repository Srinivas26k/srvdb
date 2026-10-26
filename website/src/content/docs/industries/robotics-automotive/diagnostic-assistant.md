---
title: "Diagnostic Assistant"
description: "Technicians match symptoms and trouble codes to past repairs. Level 2 Robotics & automotive project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Diagnostic Assistant"
  order: 2
---

**Technicians match symptoms and trouble codes to past repairs.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Robotics & automotive](/srvdb/industries/robotics-automotive/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Diagnostic Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Trouble codes have many causes.
- Repair history is unsearchable.
- Diagnostics take hours.

## How SrvDB powers Diagnostic Assistant

1. Repair records are embedded with codes and fixes.
2. Symptoms query similar past repairs.
3. Ranked fixes shown.
4. New repairs added with `upsert`.

## What changes

| Today | With Diagnostic Assistant on SrvDB |
|---|---|
| Trial-and-error | Ranked likely fixes |
| Lost history | Searchable repairs |
| Long diagnostics | Faster fixes |

## How it works

1. Embed repair records.
2. Query by symptom.
3. Rank fixes.
4. Repair.
5. Record outcome.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.

## Use cases

- Engine faults
- Electrical issues
- EV battery faults
- Robot actuator errors
- Fleet trucks
- Warranty analysis

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./diagnostic-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_repairs(symptom, k=5):
    hits = db.search(embed(symptom), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Technicians verify every fix.

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
| 2 · Intermediate | **Diagnostic Assistant** (this page) | Technicians match symptoms and trouble codes to past repairs. |
| 3 · Advanced | [Scenario Miner](/srvdb/industries/robotics-automotive/scenario-miner/) | Finds driving or robot episodes similar to a scenario of interest. |
| 4 · Expert | [Robot Episodic Memory](/srvdb/industries/robotics-automotive/robot-episodic-memory/) | Robots remember places, objects and outcomes. |
| 5 · Frontier | [Autonomous Fleet Memory](/srvdb/industries/robotics-automotive/autonomous-fleet-memory/) | Share learned experience across a fleet with database snapshots. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
