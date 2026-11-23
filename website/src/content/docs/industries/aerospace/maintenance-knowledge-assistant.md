---
title: "Maintenance Knowledge Assistant"
description: "Cited answers from manuals and service bulletins for technicians. Level 1 Aerospace project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Maintenance Knowledge Assistant"
  order: 1
---

**Cited answers from manuals and service bulletins for technicians.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Aerospace](/srvdb/industries/aerospace/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Maintenance Knowledge Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Manual sets are huge.
- Bulletins update procedures.
- Hangars may be offline.

## How SrvDB powers Maintenance Knowledge Assistant

1. Manuals and bulletins are embedded by task.
2. Questions return cited tasks.
3. Runs offline on tablets.
4. Revisions replace old content.

## What changes

| Today | With Maintenance Knowledge Assistant on SrvDB |
|---|---|
| Searching manuals | Cited tasks |
| Missed bulletins | Retrieved together |
| Connectivity needed | Offline |

## How it works

1. Ingest current revisions.
2. Embed.
3. Answer with citations.
4. Log usage.
5. Update per revision.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Task lookup
- Torque values
- Bulletin applicability
- Tooling
- Inspection intervals
- Training

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./maintenance-knowledge-assistant-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def find_tasks(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Official manuals remain authoritative.

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
| 1 · Beginner | **Maintenance Knowledge Assistant** (this page) | Cited answers from manuals and service bulletins for technicians. |
| 2 · Intermediate | [Safety Report Analyzer](/srvdb/industries/aerospace/safety-report-analyzer/) | Clusters safety reports and links similar events. |
| 3 · Advanced | [Fleet Finding Matcher](/srvdb/industries/aerospace/fleet-finding-matcher/) | Matches new maintenance findings to similar findings across the fleet. |
| 4 · Expert | [Borescope Image Search](/srvdb/industries/aerospace/borescope-image-search/) | Find similar borescope images and the findings they led to. |
| 5 · Frontier | [Mission Knowledge System](/srvdb/industries/aerospace/mission-knowledge-system/) | Offline mission and ground-segment knowledge for space operations. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
