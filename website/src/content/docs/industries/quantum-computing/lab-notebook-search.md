---
title: "Lab Notebook Search"
description: "Search the group's notes, papers and run summaries. Level 1 Quantum computing project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Lab Notebook Search"
  order: 1
---

**Search the group's notes, papers and run summaries.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Quantum computing](/srvdb/industries/quantum-computing/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Lab Notebook Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Run results are hard to find again.
- Notes are spread across tools.
- Unpublished work must stay private.

## How SrvDB powers Lab Notebook Search

1. Notes and run summaries are embedded locally.
2. Questions retrieve relevant entries.
3. Private to the lab.
4. Added continuously.

## What changes

| Today | With Lab Notebook Search on SrvDB |
|---|---|
| Scattered notes | One search |
| Lost runs | Found by description |
| Privacy concerns | Local only |

## How it works

1. Collect notes.
2. Embed.
3. Search.
4. Add runs.
5. Back up.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Past runs
- Calibration notes
- Paper notes
- Code snippets
- Meeting notes
- Onboarding

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./lab-notebook-search-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def find_notes(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- SrvDB is classical software running on regular computers.

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
| 1 · Beginner | **Lab Notebook Search** (this page) | Search the group's notes, papers and run summaries. |
| 2 · Intermediate | [Lab Q&A Assistant](/srvdb/industries/quantum-computing/lab-qa-assistant/) | Answers questions from the group's own knowledge with citations. |
| 3 · Advanced | [Calibration Drift Detector](/srvdb/industries/quantum-computing/calibration-drift-detector/) | Compares calibration snapshots stored as vectors to catch drift. |
| 4 · Expert | [Experiment Memory Agent](/srvdb/industries/quantum-computing/experiment-memory-agent/) | An agent remembering runs, plots and decisions across experiments. |
| 5 · Frontier | [Circuit Similarity Research](/srvdb/industries/quantum-computing/circuit-similarity-research/) | Research platform for circuit embeddings and similarity search. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
