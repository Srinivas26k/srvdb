---
title: "Hardware Lab Agent"
description: "An agent that remembers bench sessions, scope captures and decisions. Level 4 Electrical & electronics project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Hardware Lab Agent"
  order: 4
---

**An agent that remembers bench sessions, scope captures and decisions.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Electrical & electronics](/srvdb/industries/electrical-electronics/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Hardware Lab Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Bench knowledge is lost between sessions.
- Scope captures are not searchable.
- Teams repeat experiments.

## How SrvDB powers Hardware Lab Agent

1. Notes and capture images are embedded together.
2. The agent recalls past sessions by meaning.
3. One database per project.
4. Crash-safe storage.

## What changes

| Today | With Hardware Lab Agent on SrvDB |
|---|---|
| Lost bench knowledge | Persistent memory |
| Unsearchable captures | Searchable images |
| Repeated experiments | Recalled results |

## How it works

1. Log sessions.
2. Embed notes and captures.
3. Recall.
4. Summarise.
5. Archive.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Debug sessions
- Characterisation
- Compliance pre-scans
- Design reviews
- Handover
- Root-cause analysis

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./hardware-lab-agent-db", dimension=512)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def lab_recall(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Image embeddings need a suitable model.

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
| 4 · Expert | **Hardware Lab Agent** (this page) | An agent that remembers bench sessions, scope captures and decisions. |
| 5 · Frontier | [RF Fingerprint Lab](/srvdb/industries/electrical-electronics/rf-fingerprint-lab/) | Similarity search over RF spectrum fingerprints (research). |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
