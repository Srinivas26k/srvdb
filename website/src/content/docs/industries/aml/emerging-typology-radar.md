---
title: "Emerging Typology Radar"
description: "Detects clusters of cases far from every known typology. Level 5 Anti-money laundering project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Emerging Typology Radar"
  order: 5
---

**Detects clusters of cases far from every known typology.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Anti-money laundering](/srvdb/industries/aml/) | 5 of 5 | Frontier | `hnsw` |

:::note[Coming soon]
A complete, runnable **Emerging Typology Radar** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Criminals adapt faster than rules.
- New schemes look like noise.
- Detection lags.

## How SrvDB powers Emerging Typology Radar

1. Case narratives are compared with known typologies.
2. Dense clusters far from all known typologies are surfaced.
3. Analysts review and name new typologies.
4. Confidential on-prem stack.

## What changes

| Today | With Emerging Typology Radar on SrvDB |
|---|---|
| Known patterns only | New patterns surfaced |
| Detection lag | Earlier signals |
| Siloed insight | Shared new typologies |

## How it works

1. Embed cases and typologies.
2. Measure distances.
3. Find dense distant clusters.
4. Review.
5. Add new typologies.

## SrvDB features it uses

- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.

## Use cases

- New scam waves
- Novel crypto schemes
- Trade anomalies
- Mule recruitment
- Cross-border patterns
- Fintech abuse

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./emerging-typology-radar-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def nearest_typologies(case_narrative, k=5):
    hits = db.search(embed(case_narrative), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Analysts validate every new typology.

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
| 4 · Expert | [AML Investigation Agent](/srvdb/industries/aml/aml-investigation-agent/) | An investigator agent with memory across cases, typologies and media. |
| 5 · Frontier | **Emerging Typology Radar** (this page) | Detects clusters of cases far from every known typology. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
