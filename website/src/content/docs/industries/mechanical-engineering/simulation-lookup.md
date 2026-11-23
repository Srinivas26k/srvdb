---
title: "Simulation Lookup"
description: "Find the nearest previously simulated cases before running a new simulation. Level 5 Mechanical engineering project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Simulation Lookup"
  order: 5
---

**Find the nearest previously simulated cases before running a new simulation.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Mechanical engineering](/srvdb/industries/mechanical-engineering/) | 5 of 5 | Frontier | `flat` |

:::note[Coming soon]
A complete, runnable **Simulation Lookup** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Simulations are expensive.
- Past runs are rarely reused.
- Parameter spaces are large.

## How SrvDB powers Simulation Lookup

1. Simulation inputs and result summaries are stored as vectors.
2. New parameter sets retrieve the nearest past runs.
3. Engineers reuse or narrow the search.
4. Rust API in simulation pipelines.

## What changes

| Today | With Simulation Lookup on SrvDB |
|---|---|
| Re-running similar cases | Reusing nearest results |
| Lost runs | Searchable history |
| Blind exploration | Guided by neighbours |

## How it works

1. Vectorise inputs (at least 128 values).
2. Store with result summaries.
3. Query new cases.
4. Decide to reuse or run.
5. Add new runs.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.

## Use cases

- FEA reuse
- CFD case lookup
- Thermal studies
- Crash simulations
- Design-of-experiments
- Surrogate-model seeding

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./simulation-lookup-db", dimension=128)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def nearest_runs(parameters, k=5):
    hits = db.search(embed(parameters), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Neighbours are guidance, not validated results.

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
| 2 · Intermediate | [FMEA Assistant](/srvdb/industries/mechanical-engineering/fmea-assistant/) | Drafts and reviews FMEAs grounded in past FMEAs and failures. |
| 3 · Advanced | [Vibration Signature Matcher](/srvdb/industries/mechanical-engineering/vibration-signature-matcher/) | Matches vibration signatures to known fault patterns. |
| 4 · Expert | [Geometric Design Search](/srvdb/industries/mechanical-engineering/geometric-design-search/) | Search CAD models by shape using 3D shape embeddings. |
| 5 · Frontier | **Simulation Lookup** (this page) | Find the nearest previously simulated cases before running a new simulation. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
