---
title: "Experiment Memory Agent"
description: "An agent remembering runs, plots and decisions across experiments. Level 4 Quantum computing project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Experiment Memory Agent"
  order: 4
---

**An agent remembering runs, plots and decisions across experiments.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Quantum computing](/srvdb/industries/quantum-computing/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Experiment Memory Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Experiments are repeated unknowingly.
- Plots are not searchable.
- Hybrid workflows are complex.

## How SrvDB powers Experiment Memory Agent

1. Run logs and plots are embedded together.
2. The agent recalls similar past runs.
3. Pipeline indexing after each run.
4. Crash-safe.

## What changes

| Today | With Experiment Memory Agent on SrvDB |
|---|---|
| Repeated experiments | Recalled results |
| Unsearchable plots | Searchable |
| Lost decisions | Remembered |

## How it works

1. Index each run.
2. Recall similar runs.
3. Summarise.
4. Decide.
5. Record.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.

## Use cases

- Run planning
- Debugging
- Hybrid classical-quantum workflows
- Compiler pass notes
- Benchmark tracking
- Reproducibility

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./experiment-memory-agent-db", dimension=512)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_runs(summary, k=5):
    hits = db.search(embed(summary), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Plot embeddings need a suitable model.

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
| 4 · Expert | **Experiment Memory Agent** (this page) | An agent remembering runs, plots and decisions across experiments. |
| 5 · Frontier | [Circuit Similarity Research](/srvdb/industries/quantum-computing/circuit-similarity-research/) | Research platform for circuit embeddings and similarity search. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
