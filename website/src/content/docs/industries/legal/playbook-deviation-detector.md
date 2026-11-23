---
title: "Playbook Deviation Detector"
description: "Flags clauses that drift from your negotiation playbook. Level 3 Legal project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Playbook Deviation Detector"
  order: 3
---

**Flags clauses that drift from your negotiation playbook.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Legal](/srvdb/industries/legal/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Playbook Deviation Detector** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Counterparty paper hides unusual terms.
- Junior reviewers miss deviations.
- Playbooks are long.

## How SrvDB powers Playbook Deviation Detector

1. Playbook positions are embedded as the reference set.
2. Each incoming clause is matched; a low best score means a deviation.
3. Cosine thresholds are tuned per clause type.
4. Accepted fallbacks are added with `upsert`.

## What changes

| Today | With Playbook Deviation Detector on SrvDB |
|---|---|
| Line-by-line review | Deviations highlighted first |
| Inconsistent judgement | Same reference for everyone |
| Static playbooks | Learns accepted fallbacks |

## How it works

1. Embed playbook positions.
2. Split incoming contracts.
3. Score each clause.
4. Highlight low scores.
5. Record accepted positions.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- NDA review
- Supplier contracts
- SaaS terms
- Employment agreements
- Lease review
- Procurement templates

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./playbook-deviation-detector-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def nearest_position(clause, k=5):
    hits = db.search(embed(clause), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Thresholds need calibration per clause type.

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
| 3 · Advanced | **Playbook Deviation Detector** (this page) | Flags clauses that drift from your negotiation playbook. |
| 4 · Expert | [E-discovery Assistant](/srvdb/industries/legal/ediscovery-assistant/) | Deduplicates, clusters and prioritises review sets by meaning. |
| 5 · Frontier | [Air-gapped Investigation Suite](/srvdb/industries/legal/air-gapped-investigation-suite/) | Offline analysis of highly sensitive investigation documents. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
