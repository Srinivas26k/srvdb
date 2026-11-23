---
title: "Contract Q&A"
description: "Ask questions about a contract set and get answers with clause citations. Level 2 Legal project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Contract Q&A"
  order: 2
---

**Ask questions about a contract set and get answers with clause citations.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Legal](/srvdb/industries/legal/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Contract Q&A** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Reviewing large contract sets takes days.
- Answers need exact citations.
- Clients' documents are privileged.

## How SrvDB powers Contract Q&A

1. Each deal's documents get their own SrvDB database.
2. Questions retrieve clauses; a local LLM answers citing them.
3. Deleting a deal's database removes it completely.
4. `persist()` after ingestion.

## What changes

| Today | With Contract Q&A on SrvDB |
|---|---|
| Manual reading | Answers with citations in seconds |
| Notes scattered | Every answer links to its clause |
| Privilege concerns | Local only |

## How it works

1. Ingest the deal documents.
2. Embed clauses.
3. Ask questions.
4. Verify citations.
5. Archive or delete per policy.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Change-of-control questions
- Assignment restrictions
- Renewal dates
- Exclusivity
- Most-favoured-nation clauses
- Notice requirements

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./contract-qa-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def ask_contracts(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Verify every answer against the cited clause.

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
| 2 · Intermediate | **Contract Q&A** (this page) | Ask questions about a contract set and get answers with clause citations. |
| 3 · Advanced | [Playbook Deviation Detector](/srvdb/industries/legal/playbook-deviation-detector/) | Flags clauses that drift from your negotiation playbook. |
| 4 · Expert | [E-discovery Assistant](/srvdb/industries/legal/ediscovery-assistant/) | Deduplicates, clusters and prioritises review sets by meaning. |
| 5 · Frontier | [Air-gapped Investigation Suite](/srvdb/industries/legal/air-gapped-investigation-suite/) | Offline analysis of highly sensitive investigation documents. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
