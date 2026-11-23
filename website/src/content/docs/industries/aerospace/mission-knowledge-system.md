---
title: "Mission Knowledge System"
description: "Offline mission and ground-segment knowledge for space operations. Level 5 Aerospace project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Mission Knowledge System"
  order: 5
---

**Offline mission and ground-segment knowledge for space operations.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Aerospace](/srvdb/industries/aerospace/) | 5 of 5 | Frontier | `flat` |

:::note[Coming soon]
A complete, runnable **Mission Knowledge System** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Missions generate vast documentation.
- Ground stations may be isolated.
- Anomalies need fast context.

## How SrvDB powers Mission Knowledge System

1. Procedures, anomaly reports and telemetry summaries live in SrvDB.
2. Operators retrieve similar past anomalies.
3. Air-gapped and crash-safe.
4. Export-controlled data stays local.

## What changes

| Today | With Mission Knowledge System on SrvDB |
|---|---|
| Slow anomaly context | Instant similar cases |
| Network dependence | Offline |
| Scattered docs | One search |

## How it works

1. Ingest procedures and reports.
2. Embed telemetry summaries.
3. Search on anomalies.
4. Decide.
5. Record outcomes.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.

## Use cases

- Satellite anomalies
- Ground-station procedures
- Launch campaigns
- Payload operations
- Training
- Lessons learned

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./mission-knowledge-system-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_anomalies(anomaly, k=5):
    hits = db.search(embed(anomaly), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Not for flight-critical software.

:::caution
SrvDB is not certified software. Use it for decision support and knowledge retrieval, never in flight-critical or certified systems.
:::

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Aerospace projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Maintenance Knowledge Assistant](/srvdb/industries/aerospace/maintenance-knowledge-assistant/) | Cited answers from manuals and service bulletins for technicians. |
| 2 · Intermediate | [Safety Report Analyzer](/srvdb/industries/aerospace/safety-report-analyzer/) | Clusters safety reports and links similar events. |
| 3 · Advanced | [Fleet Finding Matcher](/srvdb/industries/aerospace/fleet-finding-matcher/) | Matches new maintenance findings to similar findings across the fleet. |
| 4 · Expert | [Borescope Image Search](/srvdb/industries/aerospace/borescope-image-search/) | Find similar borescope images and the findings they led to. |
| 5 · Frontier | **Mission Knowledge System** (this page) | Offline mission and ground-segment knowledge for space operations. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
