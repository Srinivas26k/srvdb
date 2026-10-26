---
title: "Circuit Similarity Research"
description: "Research platform for circuit embeddings and similarity search. Level 5 Quantum computing project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Circuit Similarity Research"
  order: 5
---

**Research platform for circuit embeddings and similarity search.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Quantum computing](/srvdb/industries/quantum-computing/) | 5 of 5 | Frontier | `flat` |

:::note[Coming soon]
A complete, runnable **Circuit Similarity Research** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Equivalent circuits are hard to spot.
- Circuit libraries grow.
- Embedding methods are experimental.

## How SrvDB powers Circuit Similarity Research

1. Circuit embeddings (from research models) are stored in SrvDB.
2. Exact `flat` search keeps experiments reproducible.
3. Rust API for pipelines.
4. Results archived with versioned formats.

## What changes

| Today | With Circuit Similarity Research on SrvDB |
|---|---|
| Manual comparison | Similarity search |
| Ad-hoc scripts | Reusable store |
| Irreproducible searches | Exact search |

## How it works

1. Choose an embedding method.
2. Embed circuits.
3. Search.
4. Evaluate.
5. Publish.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.

## Use cases

- Equivalent-circuit candidates
- Error-syndrome pattern retrieval
- Quantum chemistry result lookup
- QML experiment store
- Benchmark archives
- Teaching sets

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./circuit-similarity-research-db", dimension=256)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_circuits(circuit_description, k=5):
    hits = db.search(embed(circuit_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Research direction; results depend on the embedding method.

:::caution
SrvDB is classical software: it stores ordinary vectors describing experiments and runs on regular computers, not on quantum hardware.
:::

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Quantum computing projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Lab Notebook Search](/srvdb/industries/quantum-computing/lab-notebook-search/) | Search the group's notes, papers and run summaries. |
| 2 · Intermediate | [Lab Q&A Assistant](/srvdb/industries/quantum-computing/lab-qa-assistant/) | Answers questions from the group's own knowledge with citations. |
| 3 · Advanced | [Calibration Drift Detector](/srvdb/industries/quantum-computing/calibration-drift-detector/) | Compares calibration snapshots stored as vectors to catch drift. |
| 4 · Expert | [Experiment Memory Agent](/srvdb/industries/quantum-computing/experiment-memory-agent/) | An agent remembering runs, plots and decisions across experiments. |
| 5 · Frontier | **Circuit Similarity Research** (this page) | Research platform for circuit embeddings and similarity search. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
