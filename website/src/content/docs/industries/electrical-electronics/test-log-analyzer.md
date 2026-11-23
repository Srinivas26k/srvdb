---
title: "Test Log Analyzer"
description: "Finds similar past test failures and anomalous logs. Level 3 Electrical & electronics project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Test Log Analyzer"
  order: 3
---

**Finds similar past test failures and anomalous logs.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Electrical & electronics](/srvdb/industries/electrical-electronics/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Test Log Analyzer** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Test logs pile up unsearchable.
- Failures recur across boards.
- Root causes are re-investigated.

## How SrvDB powers Test Log Analyzer

1. Log excerpts are embedded.
2. Failures retrieve similar past failures and fixes.
3. Logs far from normal are flagged.
4. Batch processing for CI.

## What changes

| Today | With Test Log Analyzer on SrvDB |
|---|---|
| Unsearchable logs | Searchable history |
| Repeated debugging | Past fixes found |
| Silent anomalies | Flagged |

## How it works

1. Collect logs.
2. Embed.
3. Match failures.
4. Flag anomalies.
5. Record fixes.

## SrvDB features it uses

- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Board bring-up
- EMC test failures
- Production test
- Burn-in
- Firmware regressions
- Field returns

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./test-log-analyzer-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_failures(log_excerpt, k=5):
    hits = db.search(embed(log_excerpt), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Log format consistency helps.

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
| 2 · Intermediate | [Second-source Finder](/srvdb/industries/electrical-electronics/second-source-finder/) | Suggests substitute components by specification similarity. |
| 3 · Advanced | **Test Log Analyzer** (this page) | Finds similar past test failures and anomalous logs. |
| 4 · Expert | [Hardware Lab Agent](/srvdb/industries/electrical-electronics/hardware-lab-agent/) | An agent that remembers bench sessions, scope captures and decisions. |
| 5 · Frontier | [RF Fingerprint Lab](/srvdb/industries/electrical-electronics/rf-fingerprint-lab/) | Similarity search over RF spectrum fingerprints (research). |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
