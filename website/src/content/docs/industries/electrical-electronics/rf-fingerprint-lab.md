---
title: "RF Fingerprint Lab"
description: "Similarity search over RF spectrum fingerprints (research). Level 5 Electrical & electronics project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 RF Fingerprint Lab"
  order: 5
---

**Similarity search over RF spectrum fingerprints (research).**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Electrical & electronics](/srvdb/industries/electrical-electronics/) | 5 of 5 | Frontier | `sq8` |

:::note[Coming soon]
A complete, runnable **RF Fingerprint Lab** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- RF environments are crowded.
- Identifying emitters is hard.
- Data volumes are high.

## How SrvDB powers RF Fingerprint Lab

1. Spectrum features become vectors in SrvDB.
2. Unknown signals retrieve similar known fingerprints.
3. `sq8` for large libraries.
4. Offline operation in the field.

## What changes

| Today | With RF Fingerprint Lab on SrvDB |
|---|---|
| Manual spectrum analysis | Similarity lookup |
| Large raw data | Compact vectors |
| Lab-only tools | Field-capable |

## How it works

1. Extract spectral features.
2. Store labelled fingerprints.
3. Query unknowns.
4. Validate.
5. Grow the library.

## SrvDB features it uses

- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Interference hunting
- Device identification
- Spectrum monitoring
- Test labs
- IoT diagnostics
- Research datasets

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./rf-fingerprint-lab-db", dimension=128, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def known_emitters_like(fingerprint, k=5):
    hits = db.search(embed(fingerprint), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Research direction; accuracy depends on features and must be validated.

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
| 3 · Advanced | [Test Log Analyzer](/srvdb/industries/electrical-electronics/test-log-analyzer/) | Finds similar past test failures and anomalous logs. |
| 4 · Expert | [Hardware Lab Agent](/srvdb/industries/electrical-electronics/hardware-lab-agent/) | An agent that remembers bench sessions, scope captures and decisions. |
| 5 · Frontier | **RF Fingerprint Lab** (this page) | Similarity search over RF spectrum fingerprints (research). |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
