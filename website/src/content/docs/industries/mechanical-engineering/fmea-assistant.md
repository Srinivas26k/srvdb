---
title: "FMEA Assistant"
description: "Drafts and reviews FMEAs grounded in past FMEAs and failures. Level 2 Mechanical engineering project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 FMEA Assistant"
  order: 2
---

**Drafts and reviews FMEAs grounded in past FMEAs and failures.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Mechanical engineering](/srvdb/industries/mechanical-engineering/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **FMEA Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- FMEAs start from blank sheets.
- Past failure modes are forgotten.
- Quality varies by team.

## How SrvDB powers FMEA Assistant

1. Past FMEA rows and failure reports are embedded.
2. A new function retrieves relevant failure modes.
3. A local LLM drafts rows with citations.
4. Accepted rows fed back.

## What changes

| Today | With FMEA Assistant on SrvDB |
|---|---|
| Blank-sheet FMEAs | Drafts from history |
| Forgotten failures | Recalled automatically |
| Inconsistent quality | Shared knowledge |

## How it works

1. Embed past FMEAs.
2. Query per function.
3. Draft rows.
4. Engineer review.
5. Upsert accepted rows.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.

## Use cases

- Design FMEA
- Process FMEA
- Supplier FMEA
- Change reviews
- Audit preparation
- Training

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./fmea-assistant-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def failure_modes_for(function, k=5):
    hits = db.search(embed(function), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- FMEA ownership stays with engineers.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Mechanical engineering projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Part Finder](/srvdb/industries/mechanical-engineering/part-finder/) | Find existing parts by describing what they do. |
| 2 · Intermediate | **FMEA Assistant** (this page) | Drafts and reviews FMEAs grounded in past FMEAs and failures. |
| 3 · Advanced | [Vibration Signature Matcher](/srvdb/industries/mechanical-engineering/vibration-signature-matcher/) | Matches vibration signatures to known fault patterns. |
| 4 · Expert | [Geometric Design Search](/srvdb/industries/mechanical-engineering/geometric-design-search/) | Search CAD models by shape using 3D shape embeddings. |
| 5 · Frontier | [Simulation Lookup](/srvdb/industries/mechanical-engineering/simulation-lookup/) | Find the nearest previously simulated cases before running a new simulation. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
