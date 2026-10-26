---
title: "E-discovery Assistant"
description: "Deduplicates, clusters and prioritises review sets by meaning. Level 4 Legal project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 E-discovery Assistant"
  order: 4
---

**Deduplicates, clusters and prioritises review sets by meaning.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Legal](/srvdb/industries/legal/) | 4 of 5 | Expert | `hnsw` |

:::note[Coming soon]
A complete, runnable **E-discovery Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Review sets contain millions of near-duplicates.
- Privileged documents must be found early.
- Reviewers' time is expensive.

## How SrvDB powers E-discovery Assistant

1. Documents are embedded and stored with `hnsw` for scale.
2. Near-duplicates and similar documents are grouped.
3. Documents similar to known privileged ones are prioritised.
4. Per-matter databases keep sets separate.

## What changes

| Today | With E-discovery Assistant on SrvDB |
|---|---|
| Linear review | Clustered, prioritised review |
| Duplicate review effort | Near-duplicates handled once |
| Late privilege finds | Early prioritisation |

## How it works

1. Process documents to text.
2. Embed and store.
3. Cluster and deduplicate.
4. Prioritise by similarity to seeds.
5. Review and tag.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Near-duplicate suppression
- Privilege prioritisation
- Topic clustering
- Seed-set expansion
- Hot-document discovery
- Production QA

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./ediscovery-assistant-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_documents(document_text, k=5):
    hits = db.search(embed(document_text), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Check `hnsw` recall against `flat` on a sample.
- Defensibility processes remain the firm's.

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
| 4 · Expert | **E-discovery Assistant** (this page) | Deduplicates, clusters and prioritises review sets by meaning. |
| 5 · Frontier | [Air-gapped Investigation Suite](/srvdb/industries/legal/air-gapped-investigation-suite/) | Offline analysis of highly sensitive investigation documents. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
