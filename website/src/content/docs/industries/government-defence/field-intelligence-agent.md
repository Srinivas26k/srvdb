---
title: "Field Intelligence Agent"
description: "An offline agent on rugged laptops combining text and imagery. Level 4 Government & defence project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Field Intelligence Agent"
  order: 4
---

**An offline agent on rugged laptops combining text and imagery.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Government & defence](/srvdb/industries/government-defence/) | 4 of 5 | Expert | `sq8` |

:::note[Coming soon]
A complete, runnable **Field Intelligence Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Field teams lack connectivity.
- Information spans text and images.
- Power is unreliable.

## How SrvDB powers Field Intelligence Agent

1. Multimodal embeddings put text and images in SrvDB.
2. The agent answers and recalls field notes offline.
3. Crash-safe `persist()` survives power loss.
4. One database per classification level.

## What changes

| Today | With Field Intelligence Agent on SrvDB |
|---|---|
| Paper notes | Searchable field memory |
| No connectivity | Fully offline |
| Data loss on power cuts | Crash-safe storage |

## How it works

1. Preload reference data.
2. Capture notes and images.
3. Embed locally.
4. Query and recall.
5. Sync when back at base.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.

## Use cases

- Field reports
- Equipment identification by photo
- Translation memory
- Mission notes
- Logistics lookups
- After-action reviews

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./field-intelligence-agent-db", dimension=512, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def field_recall(note, k=5):
    hits = db.search(embed(note), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Handle data per classification rules.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Government & defence projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Doctrine Assistant](/srvdb/industries/government-defence/doctrine-assistant/) | Air-gapped question answering over manuals and doctrine. |
| 2 · Intermediate | [Citizen Service Bot](/srvdb/industries/government-defence/citizen-service-bot/) | Answers citizens' questions from official public guidance. |
| 3 · Advanced | [Report Linker](/srvdb/industries/government-defence/report-linker/) | Links related reports and requests across departments. |
| 4 · Expert | **Field Intelligence Agent** (this page) | An offline agent on rugged laptops combining text and imagery. |
| 5 · Frontier | [Sovereign AI Stack](/srvdb/industries/government-defence/sovereign-ai-stack/) | A fully reviewed, air-gapped AI stack with a published SBOM. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
