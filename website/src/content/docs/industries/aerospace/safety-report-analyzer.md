---
title: "Safety Report Analyzer"
description: "Clusters safety reports and links similar events. Level 2 Aerospace project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Safety Report Analyzer"
  order: 2
---

**Clusters safety reports and links similar events.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Aerospace](/srvdb/industries/aerospace/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Safety Report Analyzer** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Safety reports arrive as free text.
- Similar events are scattered.
- Trends emerge slowly.

## How SrvDB powers Safety Report Analyzer

1. Reports are embedded and linked to neighbours.
2. Clusters reveal trends.
3. New reports far from known clusters are flagged.
4. Batch processing.

## What changes

| Today | With Safety Report Analyzer on SrvDB |
|---|---|
| Manual reading | Pre-grouped reports |
| Scattered events | Linked events |
| Slow trends | Early clusters |

## How it works

1. Ingest reports.
2. Embed.
3. Cluster.
4. Review trends.
5. Act.

## SrvDB features it uses

- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Ground incidents
- Maintenance errors
- Bird strikes
- Cabin events
- Ramp safety
- Training findings

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./safety-report-analyzer-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_events(report, k=5):
    hits = db.search(embed(report), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Safety analysts make determinations.

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
| 2 · Intermediate | **Safety Report Analyzer** (this page) | Clusters safety reports and links similar events. |
| 3 · Advanced | [Fleet Finding Matcher](/srvdb/industries/aerospace/fleet-finding-matcher/) | Matches new maintenance findings to similar findings across the fleet. |
| 4 · Expert | [Borescope Image Search](/srvdb/industries/aerospace/borescope-image-search/) | Find similar borescope images and the findings they led to. |
| 5 · Frontier | [Mission Knowledge System](/srvdb/industries/aerospace/mission-knowledge-system/) | Offline mission and ground-segment knowledge for space operations. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
