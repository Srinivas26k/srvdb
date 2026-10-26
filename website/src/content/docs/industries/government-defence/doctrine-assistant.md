---
title: "Doctrine Assistant"
description: "Air-gapped question answering over manuals and doctrine. Level 1 Government & defence project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Doctrine Assistant"
  order: 1
---

**Air-gapped question answering over manuals and doctrine.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Government & defence](/srvdb/industries/government-defence/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Doctrine Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Manuals run to thousands of pages.
- Networks are isolated.
- Answers must cite sources.

## How SrvDB powers Doctrine Assistant

1. Manuals are embedded on an isolated machine.
2. Questions retrieve passages; a local LLM answers with citations.
3. No network access at all.
4. Updates via `upsert`.

## What changes

| Today | With Doctrine Assistant on SrvDB |
|---|---|
| Manual lookup | Cited answers in seconds |
| No AI on isolated networks | Fully offline AI |
| Inconsistent interpretations | Shared sources |

## How it works

1. Import documents.
2. Embed locally.
3. Answer with citations.
4. Log usage.
5. Update with new editions.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Procedure lookup
- Equipment manuals
- Training support
- Policy questions
- Standard operating procedures
- Exercise preparation

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./doctrine-assistant-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def cite(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Content classification rules apply to the database folder.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Government & defence projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Doctrine Assistant** (this page) | Air-gapped question answering over manuals and doctrine. |
| 2 · Intermediate | [Citizen Service Bot](/srvdb/industries/government-defence/citizen-service-bot/) | Answers citizens' questions from official public guidance. |
| 3 · Advanced | [Report Linker](/srvdb/industries/government-defence/report-linker/) | Links related reports and requests across departments. |
| 4 · Expert | [Field Intelligence Agent](/srvdb/industries/government-defence/field-intelligence-agent/) | An offline agent on rugged laptops combining text and imagery. |
| 5 · Frontier | [Sovereign AI Stack](/srvdb/industries/government-defence/sovereign-ai-stack/) | A fully reviewed, air-gapped AI stack with a published SBOM. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
