---
title: "Defect Matcher"
description: "Matches new defects to past ones, from text and images. Level 2 Manufacturing & IoT project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Defect Matcher"
  order: 2
---

**Matches new defects to past ones, from text and images.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Manufacturing & IoT](/srvdb/industries/manufacturing-iot/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Defect Matcher** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Same defects recur on different lines.
- Root causes are re-investigated.
- Images are not searchable.

## How SrvDB powers Defect Matcher

1. Defect reports and photos are embedded (multimodal).
2. New defects retrieve the closest past ones with root causes.
3. Cosine thresholds flag recurring defects.
4. 8D reports added with `upsert`.

## What changes

| Today | With Defect Matcher on SrvDB |
|---|---|
| Repeated investigations | Past root causes in seconds |
| Images unused | Image similarity search |
| Line silos | Plant-wide matches |

## How it works

1. Embed reports and images.
2. Query with a new defect.
3. Review matches.
4. Confirm root cause.
5. Upsert the outcome.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Surface defects
- Weld quality issues
- Assembly errors
- Supplier non-conformances
- Customer returns
- Warranty claims

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./defect-matcher-db", dimension=512)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_defects(defect_description, k=5):
    hits = db.search(embed(defect_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Image similarity needs a suitable vision embedding model.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Manufacturing & IoT projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Maintenance Assistant](/srvdb/industries/manufacturing-iot/maintenance-assistant/) | Technicians describe a symptom and get the right manual section and past fixes. |
| 2 · Intermediate | **Defect Matcher** (this page) | Matches new defects to past ones, from text and images. |
| 3 · Advanced | [Sensor Anomaly Detector](/srvdb/industries/manufacturing-iot/sensor-anomaly-detector/) | Flags sensor windows far from normal operation. |
| 4 · Expert | [Plant Knowledge Agent](/srvdb/industries/manufacturing-iot/plant-knowledge-agent/) | An agent with shift memory across lines, manuals and quality records. |
| 5 · Frontier | [Predictive Maintenance at the Edge](/srvdb/industries/manufacturing-iot/predictive-maintenance-edge/) | Finds machines whose behaviour resembles past pre-failure signatures. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
