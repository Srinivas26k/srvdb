---
title: "Adverse Media Matcher"
description: "Deduplicates and matches adverse media to customers. Level 3 Anti-money laundering project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Adverse Media Matcher"
  order: 3
---

**Deduplicates and matches adverse media to customers.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Anti-money laundering](/srvdb/industries/aml/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Adverse Media Matcher** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Adverse media volumes are huge.
- Many articles repeat the same story.
- Matching names is error-prone.

## How SrvDB powers Adverse Media Matcher

1. Articles are embedded and deduplicated.
2. Customer profiles retrieve relevant articles.
3. Multilingual embeddings span languages.
4. Results reviewed by analysts.

## What changes

| Today | With Adverse Media Matcher on SrvDB |
|---|---|
| Duplicate articles | Deduplicated stories |
| Name-only matching | Context matching |
| Language gaps | Multilingual |

## How it works

1. Ingest media.
2. Embed and deduplicate.
3. Match profiles.
4. Review.
5. Record decisions.

## SrvDB features it uses

- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Onboarding screening
- Periodic reviews
- PEP context
- Sanctions context
- Supplier screening
- Investigations

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./adverse-media-matcher-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def relevant_articles(profile, k=5):
    hits = db.search(embed(profile), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Supplements, does not replace, screening systems.

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
| 3 · Advanced | **Adverse Media Matcher** (this page) | Deduplicates and matches adverse media to customers. |
| 4 · Expert | [AML Investigation Agent](/srvdb/industries/aml/aml-investigation-agent/) | An investigator agent with memory across cases, typologies and media. |
| 5 · Frontier | [Emerging Typology Radar](/srvdb/industries/aml/emerging-typology-radar/) | Detects clusters of cases far from every known typology. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
