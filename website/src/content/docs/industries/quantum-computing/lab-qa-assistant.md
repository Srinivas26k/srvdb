---
title: "Lab Q&A Assistant"
description: "Answers questions from the group's own knowledge with citations. Level 2 Quantum computing project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Lab Q&A Assistant"
  order: 2
---

**Answers questions from the group's own knowledge with citations.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Quantum computing](/srvdb/industries/quantum-computing/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Lab Q&A Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- New members ask the same questions.
- Knowledge sits with senior researchers.
- Generic chatbots lack context.

## How SrvDB powers Lab Q&A Assistant

1. Lab knowledge is embedded.
2. A local LLM answers with citations.
3. Updated as work progresses.
4. Private.

## What changes

| Today | With Lab Q&A Assistant on SrvDB |
|---|---|
| Interrupting seniors | Cited answers |
| Generic answers | Lab-specific answers |
| Lost knowledge | Captured knowledge |

## How it works

1. Ingest knowledge.
2. Ask.
3. Verify sources.
4. Update.
5. Review.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Hardware procedures
- Calibration steps
- Software setup
- Algorithm notes
- Experiment design
- Write-ups

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./lab-qa-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def lab_answer(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Answers cite sources; researchers verify.

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
| 2 · Intermediate | **Lab Q&A Assistant** (this page) | Answers questions from the group's own knowledge with citations. |
| 3 · Advanced | [Calibration Drift Detector](/srvdb/industries/quantum-computing/calibration-drift-detector/) | Compares calibration snapshots stored as vectors to catch drift. |
| 4 · Expert | [Experiment Memory Agent](/srvdb/industries/quantum-computing/experiment-memory-agent/) | An agent remembering runs, plots and decisions across experiments. |
| 5 · Frontier | [Circuit Similarity Research](/srvdb/industries/quantum-computing/circuit-similarity-research/) | Research platform for circuit embeddings and similarity search. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
