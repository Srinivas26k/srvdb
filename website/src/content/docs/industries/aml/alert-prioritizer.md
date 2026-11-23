---
title: "Alert Prioritizer"
description: "Ranks AML alerts by similarity to past true and false positives. Level 2 Anti-money laundering project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Alert Prioritizer"
  order: 2
---

**Ranks AML alerts by similarity to past true and false positives.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Anti-money laundering](/srvdb/industries/aml/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Alert Prioritizer** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Most alerts are false positives.
- Investigators drown in queues.
- Past dispositions are unused.

## How SrvDB powers Alert Prioritizer

1. Past alert narratives are embedded with dispositions.
2. New alerts retrieve neighbours and their outcomes.
3. Queues are ordered by risk signal.
4. Dispositions fed back.

## What changes

| Today | With Alert Prioritizer on SrvDB |
|---|---|
| FIFO queues | Prioritised queues |
| Unused history | Learning from dispositions |
| Burnout | Focus on likely cases |

## How it works

1. Embed past alerts.
2. Score new alerts.
3. Rank.
4. Investigate.
5. Feed back.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Transaction monitoring
- Sanctions alerts
- Name-screening alerts
- Trade surveillance
- Card alerts
- Correspondent banking

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./alert-prioritizer-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_cases(narrative, k=5):
    hits = db.search(embed(narrative), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Decision support only; all alerts still require proper handling.

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
| 2 · Intermediate | **Alert Prioritizer** (this page) | Ranks AML alerts by similarity to past true and false positives. |
| 3 · Advanced | [Adverse Media Matcher](/srvdb/industries/aml/adverse-media-matcher/) | Deduplicates and matches adverse media to customers. |
| 4 · Expert | [AML Investigation Agent](/srvdb/industries/aml/aml-investigation-agent/) | An investigator agent with memory across cases, typologies and media. |
| 5 · Frontier | [Emerging Typology Radar](/srvdb/industries/aml/emerging-typology-radar/) | Detects clusters of cases far from every known typology. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
