---
title: "Scenario Miner"
description: "Finds driving or robot episodes similar to a scenario of interest. Level 3 Robotics & automotive project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Scenario Miner"
  order: 3
---

**Finds driving or robot episodes similar to a scenario of interest.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Robotics & automotive](/srvdb/industries/robotics-automotive/) | 3 of 5 | Advanced | `hnsw` |

:::note[Coming soon]
A complete, runnable **Scenario Miner** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Fleet data is enormous.
- Rare scenarios are hard to find.
- Testing needs edge cases.

## How SrvDB powers Scenario Miner

1. Episodes are summarised as embeddings (scene, sensor features).
2. Example scenarios query similar episodes.
3. `hnsw` and `sq8` for scale.
4. Batch search over new data.

## What changes

| Today | With Scenario Miner on SrvDB |
|---|---|
| Manual video review | Similarity search |
| Rare cases missed | Found by example |
| Huge storage | Compressed vectors |

## How it works

1. Embed episodes.
2. Store with metadata.
3. Query by example.
4. Curate test sets.
5. Repeat on new data.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Edge-case mining
- Near-miss analysis
- Test-set curation
- Sensor fault episodes
- Weather conditions
- Robot grasp failures

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./scenario-miner-db", dimension=256, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_episodes(scenario, k=5):
    hits = db.search(embed(scenario), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Quality depends on scene embeddings.

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
| 3 · Advanced | **Scenario Miner** (this page) | Finds driving or robot episodes similar to a scenario of interest. |
| 4 · Expert | [Robot Episodic Memory](/srvdb/industries/robotics-automotive/robot-episodic-memory/) | Robots remember places, objects and outcomes. |
| 5 · Frontier | [Autonomous Fleet Memory](/srvdb/industries/robotics-automotive/autonomous-fleet-memory/) | Share learned experience across a fleet with database snapshots. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
