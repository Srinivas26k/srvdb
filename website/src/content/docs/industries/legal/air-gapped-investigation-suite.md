---
title: "Air-gapped Investigation Suite"
description: "Offline analysis of highly sensitive investigation documents. Level 5 Legal project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Air-gapped Investigation Suite"
  order: 5
---

**Offline analysis of highly sensitive investigation documents.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Legal](/srvdb/industries/legal/) | 5 of 5 | Frontier | `flat` |

:::note[Coming soon]
A complete, runnable **Air-gapped Investigation Suite** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Some investigations cannot touch any network.
- Teams need modern search anyway.
- Evidence handling must be auditable.

## How SrvDB powers Air-gapped Investigation Suite

1. SrvDB, a local embedding model and a local LLM run on one isolated machine.
2. Every retrieval is logged locally.
3. Crash-safe storage protects evidence indexes.
4. Multilingual embeddings link documents across languages.

## What changes

| Today | With Air-gapped Investigation Suite on SrvDB |
|---|---|
| Keyword search on isolated laptops | Semantic search fully offline |
| No AI for sensitive matters | AI with zero connectivity |
| Manual evidence logs | Logged retrievals |

## How it works

1. Install from reviewed wheels.
2. Ingest evidence.
3. Search and question.
4. Log retrievals.
5. Archive securely.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.

## Use cases

- Internal investigations
- Regulatory inquiries
- Whistleblower material
- Cross-border document sets
- Fact chronology building
- Interview preparation

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./air-gapped-investigation-suite-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def search_evidence(query, k=5):
    hits = db.search(embed(query), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Chain-of-custody procedures remain your responsibility.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Legal projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Clause Finder](/srvdb/industries/legal/clause-finder/) | Finds the closest clauses your firm has already used, across every contract. |
| 2 · Intermediate | [Contract Q&A](/srvdb/industries/legal/contract-qa/) | Ask questions about a contract set and get answers with clause citations. |
| 3 · Advanced | [Playbook Deviation Detector](/srvdb/industries/legal/playbook-deviation-detector/) | Flags clauses that drift from your negotiation playbook. |
| 4 · Expert | [E-discovery Assistant](/srvdb/industries/legal/ediscovery-assistant/) | Deduplicates, clusters and prioritises review sets by meaning. |
| 5 · Frontier | **Air-gapped Investigation Suite** (this page) | Offline analysis of highly sensitive investigation documents. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
