---
title: "Geometric Design Search"
description: "Search CAD models by shape using 3D shape embeddings. Level 4 Mechanical engineering project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Geometric Design Search"
  order: 4
---

**Search CAD models by shape using 3D shape embeddings.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Mechanical engineering](/srvdb/industries/mechanical-engineering/) | 4 of 5 | Expert | `hnsw` |

:::note[Coming soon]
A complete, runnable **Geometric Design Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Similar shapes hide behind different names.
- CAD search is metadata-only.
- Reuse needs geometry.

## How SrvDB powers Geometric Design Search

1. A 3D shape embedding model turns models into vectors stored in SrvDB.
2. Query by an example model.
3. `hnsw` scales to large libraries.
4. Per-programme databases.

## What changes

| Today | With Geometric Design Search on SrvDB |
|---|---|
| Metadata search | Shape search |
| Duplicate geometry | Found by similarity |
| Programme silos | Searchable per programme |

## How it works

1. Export meshes.
2. Embed with a shape model.
3. Store.
4. Query by example.
5. Review.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.

## Use cases

- Casting reuse
- Tooling reuse
- Supplier quoting
- Duplicate detection
- Design inspiration
- Cost estimation

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./geometric-design-search-db", dimension=512, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_shapes(model_description, k=5):
    hits = db.search(embed(model_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Needs a suitable 3D shape embedding model (research-grade).

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
| 4 · Expert | **Geometric Design Search** (this page) | Search CAD models by shape using 3D shape embeddings. |
| 5 · Frontier | [Simulation Lookup](/srvdb/industries/mechanical-engineering/simulation-lookup/) | Find the nearest previously simulated cases before running a new simulation. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
