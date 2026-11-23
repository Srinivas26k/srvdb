---
title: "Part Finder"
description: "Find existing parts by describing what they do. Level 1 Mechanical engineering project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Part Finder"
  order: 1
---

**Find existing parts by describing what they do.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Mechanical engineering](/srvdb/industries/mechanical-engineering/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Part Finder** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Engineers redesign parts that already exist.
- Part numbers say nothing about function.
- PLM search is keyword-based.

## How SrvDB powers Part Finder

1. Part descriptions and drawing notes are embedded in SrvDB.
2. A functional description finds candidate parts.
3. Scores rank how close each is.
4. New parts added with `upsert`.

## What changes

| Today | With Part Finder on SrvDB |
|---|---|
| New part numbers for old parts | Reuse existing parts |
| Keyword PLM search | Search by function |
| Part proliferation | Fewer duplicates |

## How it works

1. Export part metadata.
2. Embed descriptions.
3. Search by requirement.
4. Review candidates.
5. Add new parts.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.

## Use cases

- Bracket and fastener reuse
- Standard-part selection
- Supplier part lookup
- Cost reduction
- Design reviews
- Obsolescence replacement

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./part-finder-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def existing_parts_like(requirement, k=5):
    hits = db.search(embed(requirement), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Engineers verify fit, form and function.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Mechanical engineering projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Part Finder** (this page) | Find existing parts by describing what they do. |
| 2 · Intermediate | [FMEA Assistant](/srvdb/industries/mechanical-engineering/fmea-assistant/) | Drafts and reviews FMEAs grounded in past FMEAs and failures. |
| 3 · Advanced | [Vibration Signature Matcher](/srvdb/industries/mechanical-engineering/vibration-signature-matcher/) | Matches vibration signatures to known fault patterns. |
| 4 · Expert | [Geometric Design Search](/srvdb/industries/mechanical-engineering/geometric-design-search/) | Search CAD models by shape using 3D shape embeddings. |
| 5 · Frontier | [Simulation Lookup](/srvdb/industries/mechanical-engineering/simulation-lookup/) | Find the nearest previously simulated cases before running a new simulation. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
