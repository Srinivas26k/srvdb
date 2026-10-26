---
title: "Sovereign AI Stack"
description: "A fully reviewed, air-gapped AI stack with a published SBOM. Level 5 Government & defence project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Sovereign AI Stack"
  order: 5
---

**A fully reviewed, air-gapped AI stack with a published SBOM.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Government & defence](/srvdb/industries/government-defence/) | 5 of 5 | Frontier | `hnsw` |

:::note[Coming soon]
A complete, runnable **Sovereign AI Stack** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Supply-chain assurance is mandatory.
- Cloud AI is not an option.
- Systems must run for years.

## How SrvDB powers Sovereign AI Stack

1. SrvDB has no Python runtime dependencies and ships an SBOM with every release.
2. Builds from reviewed source; Apache-2.0 licensed.
3. Versioned on-disk formats support long-lived archives.
4. Crash-safe storage.

## What changes

| Today | With Sovereign AI Stack on SrvDB |
|---|---|
| Opaque AI dependencies | Reviewable components |
| Vendor lock-in | Open source |
| Short-lived prototypes | Long-lived systems |

## How it works

1. Review source and SBOM.
2. Build internally.
3. Deploy offline.
4. Operate with local models.
5. Audit regularly.

## SrvDB features it uses

- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.

## Use cases

- National archives search
- Defence knowledge bases
- Secure translation
- Intelligence linking
- Disaster-response knowledge
- Ship and vehicle deployments

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./sovereign-ai-stack-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def retrieve(query, k=5):
    hits = db.search(embed(query), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Accreditation is the operator's responsibility.

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
| 3 · Advanced | [Report Linker](/srvdb/industries/government-defence/report-linker/) | Links related reports and requests across departments. |
| 4 · Expert | [Field Intelligence Agent](/srvdb/industries/government-defence/field-intelligence-agent/) | An offline agent on rugged laptops combining text and imagery. |
| 5 · Frontier | **Sovereign AI Stack** (this page) | A fully reviewed, air-gapped AI stack with a published SBOM. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
