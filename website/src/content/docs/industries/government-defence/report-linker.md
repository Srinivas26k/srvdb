---
title: "Report Linker"
description: "Links related reports and requests across departments. Level 3 Government & defence project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Report Linker"
  order: 3
---

**Links related reports and requests across departments.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Government & defence](/srvdb/industries/government-defence/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Report Linker** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Duplicate requests waste effort.
- Related reports sit in different units.
- Patterns are missed.

## How SrvDB powers Report Linker

1. Reports are embedded and linked to nearest neighbours.
2. Duplicates are flagged.
3. Clusters reveal patterns.
4. `search_batch` for daily volumes.

## What changes

| Today | With Report Linker on SrvDB |
|---|---|
| Siloed reports | Linked across units |
| Duplicate work | Duplicates flagged |
| Patterns missed | Clusters surface |

## How it works

1. Ingest reports.
2. Embed.
3. Link neighbours.
4. Review clusters.
5. Act on duplicates.

## SrvDB features it uses

- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Information-request duplicates
- Inspection report linking
- Maintenance log patterns
- Public disinformation narratives
- Procurement bid similarity for review
- OSINT deduplication

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./report-linker-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def related_reports(report, k=5):
    hits = db.search(embed(report), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Signals need human analysis.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Government & defence projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Doctrine Assistant](/srvdb/industries/government-defence/doctrine-assistant/) | Air-gapped question answering over manuals and doctrine. |
| 2 · Intermediate | [Citizen Service Bot](/srvdb/industries/government-defence/citizen-service-bot/) | Answers citizens' questions from official public guidance. |
| 3 · Advanced | **Report Linker** (this page) | Links related reports and requests across departments. |
| 4 · Expert | [Field Intelligence Agent](/srvdb/industries/government-defence/field-intelligence-agent/) | An offline agent on rugged laptops combining text and imagery. |
| 5 · Frontier | [Sovereign AI Stack](/srvdb/industries/government-defence/sovereign-ai-stack/) | A fully reviewed, air-gapped AI stack with a published SBOM. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
