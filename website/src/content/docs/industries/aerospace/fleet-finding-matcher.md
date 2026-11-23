---
title: "Fleet Finding Matcher"
description: "Matches new maintenance findings to similar findings across the fleet. Level 3 Aerospace project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Fleet Finding Matcher"
  order: 3
---

**Matches new maintenance findings to similar findings across the fleet.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Aerospace](/srvdb/industries/aerospace/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Fleet Finding Matcher** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Findings on one aircraft recur on others.
- Records are siloed.
- Fixes are rediscovered.

## How SrvDB powers Fleet Finding Matcher

1. Findings are embedded with tail number and fix.
2. New findings retrieve similar fleet findings.
3. Snapshots shared between bases.
4. Updated continuously.

## What changes

| Today | With Fleet Finding Matcher on SrvDB |
|---|---|
| Per-aircraft records | Fleet view |
| Rediscovered fixes | Known fixes |
| Base silos | Shared snapshots |

## How it works

1. Embed findings.
2. Match new ones.
3. Show fixes.
4. Share snapshots.
5. Track trends.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.

## Use cases

- Corrosion findings
- Avionics faults
- Engine findings
- Structural issues
- Cabin defects
- Repeat defects

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./fleet-finding-matcher-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_findings(defect, k=5):
    hits = db.search(embed(defect), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Engineering approves actions.

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
| 3 · Advanced | **Fleet Finding Matcher** (this page) | Matches new maintenance findings to similar findings across the fleet. |
| 4 · Expert | [Borescope Image Search](/srvdb/industries/aerospace/borescope-image-search/) | Find similar borescope images and the findings they led to. |
| 5 · Frontier | [Mission Knowledge System](/srvdb/industries/aerospace/mission-knowledge-system/) | Offline mission and ground-segment knowledge for space operations. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
