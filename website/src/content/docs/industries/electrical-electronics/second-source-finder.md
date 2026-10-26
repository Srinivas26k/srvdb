---
title: "Second-source Finder"
description: "Suggests substitute components by specification similarity. Level 2 Electrical & electronics project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Second-source Finder"
  order: 2
---

**Suggests substitute components by specification similarity.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Electrical & electronics](/srvdb/industries/electrical-electronics/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Second-source Finder** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Shortages stop production.
- Substitute search is manual.
- Specs are described inconsistently.

## How SrvDB powers Second-source Finder

1. Component specifications are embedded.
2. A shortage part queries similar candidates.
3. Scores rank closeness.
4. Engineers verify.

## What changes

| Today | With Second-source Finder on SrvDB |
|---|---|
| Manual distributor search | Ranked candidates |
| Production stops | Faster substitutes |
| Inconsistent specs | Meaning-based matching |

## How it works

1. Embed spec descriptions.
2. Query with the short part.
3. Rank.
4. Verify parameters.
5. Record approvals.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Regulators
- MOSFETs
- Connectors
- Passives
- MCUs
- Sensors

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./second-source-finder-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def substitutes_for(part_spec, k=5):
    hits = db.search(embed(part_spec), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Electrical and footprint compatibility must be verified.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Electrical & electronics projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Datasheet Q&A](/srvdb/industries/electrical-electronics/datasheet-qa/) | Ask questions of your component datasheets and get page citations. |
| 2 · Intermediate | **Second-source Finder** (this page) | Suggests substitute components by specification similarity. |
| 3 · Advanced | [Test Log Analyzer](/srvdb/industries/electrical-electronics/test-log-analyzer/) | Finds similar past test failures and anomalous logs. |
| 4 · Expert | [Hardware Lab Agent](/srvdb/industries/electrical-electronics/hardware-lab-agent/) | An agent that remembers bench sessions, scope captures and decisions. |
| 5 · Frontier | [RF Fingerprint Lab](/srvdb/industries/electrical-electronics/rf-fingerprint-lab/) | Similarity search over RF spectrum fingerprints (research). |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
