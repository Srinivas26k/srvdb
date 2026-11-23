---
title: "Typology Library"
description: "Search money-laundering typologies and red flags by description. Level 1 Anti-money laundering project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Typology Library"
  order: 1
---

**Search money-laundering typologies and red flags by description.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Anti-money laundering](/srvdb/industries/aml/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Typology Library** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Typology knowledge is spread across reports.
- New investigators lack experience.
- Red flags are described differently.

## How SrvDB powers Typology Library

1. Typologies and red-flag descriptions are embedded.
2. Case descriptions retrieve matching typologies.
3. Local and confidential.
4. Updated with new guidance.

## What changes

| Today | With Typology Library on SrvDB |
|---|---|
| Searching reports | Instant typologies |
| Experience gaps | Shared knowledge |
| Inconsistent terms | Meaning-based |

## How it works

1. Collect typologies.
2. Embed.
3. Search.
4. Train staff.
5. Update.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Trade-based laundering
- Structuring
- Shell companies
- Crypto patterns
- Real-estate laundering
- Mule networks

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./typology-library-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def matching_typologies(case_description, k=5):
    hits = db.search(embed(case_description), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Use alongside regulatory guidance.

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
| 1 · Beginner | **Typology Library** (this page) | Search money-laundering typologies and red flags by description. |
| 2 · Intermediate | [Alert Prioritizer](/srvdb/industries/aml/alert-prioritizer/) | Ranks AML alerts by similarity to past true and false positives. |
| 3 · Advanced | [Adverse Media Matcher](/srvdb/industries/aml/adverse-media-matcher/) | Deduplicates and matches adverse media to customers. |
| 4 · Expert | [AML Investigation Agent](/srvdb/industries/aml/aml-investigation-agent/) | An investigator agent with memory across cases, typologies and media. |
| 5 · Frontier | [Emerging Typology Radar](/srvdb/industries/aml/emerging-typology-radar/) | Detects clusters of cases far from every known typology. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
