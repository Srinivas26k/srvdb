---
title: "AML Investigation Agent"
description: "An investigator agent with memory across cases, typologies and media. Level 4 Anti-money laundering project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 AML Investigation Agent"
  order: 4
---

**An investigator agent with memory across cases, typologies and media.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Anti-money laundering](/srvdb/industries/aml/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **AML Investigation Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Investigations span many systems.
- Narratives take hours.
- Context is lost between investigators.

## How SrvDB powers AML Investigation Agent

1. Databases for cases, typologies and adverse media.
2. The agent gathers evidence per case.
3. Drafts narratives with citations.
4. Case memory persists.

## What changes

| Today | With AML Investigation Agent on SrvDB |
|---|---|
| Manual evidence gathering | Automatic context |
| Hours of writing | Drafted narratives |
| Lost context | Persistent memory |

## How it works

1. Populate databases.
2. Agent gathers evidence.
3. Draft narrative.
4. Investigator reviews.
5. Store memory.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Case investigations
- Report drafting
- Quality assurance
- Look-backs
- Regulator requests
- Training

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./aml-investigation-agent-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def case_context(case_summary, k=5):
    hits = db.search(embed(case_summary), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Investigators sign off every report.

:::caution
Decision support only: SrvDB does not replace regulated screening systems or investigator judgement.
:::

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Anti-money laundering projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Typology Library](/srvdb/industries/aml/typology-library/) | Search money-laundering typologies and red flags by description. |
| 2 · Intermediate | [Alert Prioritizer](/srvdb/industries/aml/alert-prioritizer/) | Ranks AML alerts by similarity to past true and false positives. |
| 3 · Advanced | [Adverse Media Matcher](/srvdb/industries/aml/adverse-media-matcher/) | Deduplicates and matches adverse media to customers. |
| 4 · Expert | **AML Investigation Agent** (this page) | An investigator agent with memory across cases, typologies and media. |
| 5 · Frontier | [Emerging Typology Radar](/srvdb/industries/aml/emerging-typology-radar/) | Detects clusters of cases far from every known typology. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
