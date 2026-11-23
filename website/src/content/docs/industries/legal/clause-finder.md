---
title: "Clause Finder"
description: "Finds the closest clauses your firm has already used, across every contract. Level 1 Legal project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Clause Finder"
  order: 1
---

**Finds the closest clauses your firm has already used, across every contract.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Legal](/srvdb/industries/legal/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Clause Finder** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Lawyers re-draft clauses that exist somewhere in the archive.
- Keyword search misses differently worded clauses.
- Archives are confidential.

## How SrvDB powers Clause Finder

1. Every clause is embedded and stored in SrvDB with contract, client and date metadata.
2. Drafting a clause is the query; the closest precedents come back ranked.
3. Stays on the firm's own machines.
4. New executed contracts are added with `upsert`.

## What changes

| Today | With Clause Finder on SrvDB |
|---|---|
| Searching folders for a precedent | Precedents appear while drafting |
| Inconsistent wording | Firm-standard wording is easy to reuse |
| Confidential archives off-limits to AI | All local |

## How it works

1. Split contracts into clauses.
2. Embed and store with metadata.
3. Query with the draft clause.
4. Review precedents.
5. Add executed contracts.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Limitation-of-liability precedents
- Termination clauses
- Governing-law variants
- Data-protection terms
- Indemnities
- Client-specific preferred wording

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./clause-finder-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def closest_clauses(draft_clause, k=5):
    hits = db.search(embed(draft_clause), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Lawyers must review every suggestion.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Legal projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Clause Finder** (this page) | Finds the closest clauses your firm has already used, across every contract. |
| 2 · Intermediate | [Contract Q&A](/srvdb/industries/legal/contract-qa/) | Ask questions about a contract set and get answers with clause citations. |
| 3 · Advanced | [Playbook Deviation Detector](/srvdb/industries/legal/playbook-deviation-detector/) | Flags clauses that drift from your negotiation playbook. |
| 4 · Expert | [E-discovery Assistant](/srvdb/industries/legal/ediscovery-assistant/) | Deduplicates, clusters and prioritises review sets by meaning. |
| 5 · Frontier | [Air-gapped Investigation Suite](/srvdb/industries/legal/air-gapped-investigation-suite/) | Offline analysis of highly sensitive investigation documents. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
