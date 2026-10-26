---
title: "Vibration Signature Matcher"
description: "Matches vibration signatures to known fault patterns. Level 3 Mechanical engineering project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Vibration Signature Matcher"
  order: 3
---

**Matches vibration signatures to known fault patterns.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Mechanical engineering](/srvdb/industries/mechanical-engineering/) | 3 of 5 | Advanced | `sq8` |

:::note[Coming soon]
A complete, runnable **Vibration Signature Matcher** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Vibration analysis needs experts.
- Fault libraries are tribal knowledge.
- Monitoring data is huge.

## How SrvDB powers Vibration Signature Matcher

1. Spectra are converted to feature vectors and stored in SrvDB.
2. New measurements retrieve similar known faults.
3. Distance from healthy signatures flags anomalies.
4. `sq8` keeps libraries small.

## What changes

| Today | With Vibration Signature Matcher on SrvDB |
|---|---|
| Expert-only analysis | Assisted matching |
| Tribal fault libraries | Shared library |
| Data overload | Compact vectors |

## How it works

1. Compute spectral features (at least 128 values).
2. Store labelled signatures.
3. Match new data.
4. Review.
5. Grow the library.

## SrvDB features it uses

- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Bearing faults
- Imbalance
- Misalignment
- Gear wear
- Looseness
- Resonance

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./vibration-signature-matcher-db", dimension=128, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def known_faults_like(signature, k=5):
    hits = db.search(embed(signature), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Feature engineering drives accuracy.

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
| 3 · Advanced | **Vibration Signature Matcher** (this page) | Matches vibration signatures to known fault patterns. |
| 4 · Expert | [Geometric Design Search](/srvdb/industries/mechanical-engineering/geometric-design-search/) | Search CAD models by shape using 3D shape embeddings. |
| 5 · Frontier | [Simulation Lookup](/srvdb/industries/mechanical-engineering/simulation-lookup/) | Find the nearest previously simulated cases before running a new simulation. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
