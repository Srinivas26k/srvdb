---
title: "Investigation Workbench"
description: "Isolated, multimodal workspace per investigation. Level 4 Media & journalism project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Investigation Workbench"
  order: 4
---

**Isolated, multimodal workspace per investigation.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Media & journalism](/srvdb/industries/media-journalism/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Investigation Workbench** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Investigations mix documents, images and transcripts.
- Sources need protection.
- Teams collaborate under pressure.

## How SrvDB powers Investigation Workbench

1. One SrvDB database per investigation.
2. Text and images embedded together.
3. Runs on the newsroom's own hardware.
4. Delete the folder to remove everything.

## What changes

| Today | With Investigation Workbench on SrvDB |
|---|---|
| Scattered materials | One searchable workspace |
| Source exposure risk | Local only |
| Messy cleanup | Delete one folder |

## How it works

1. Create a workspace.
2. Ingest materials.
3. Search and annotate.
4. Draft.
5. Archive or delete.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Document leaks
- Court records
- Interview transcripts
- Photo verification
- Timeline building
- Podcast research

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./investigation-workbench-db", dimension=512)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def investigation_search(query, k=5):
    hits = db.search(embed(query), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Follow source-protection procedures.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Media & journalism projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Archive Search](/srvdb/industries/media-journalism/archive-search/) | Search decades of articles by meaning. |
| 2 · Intermediate | [Fact-check Assistant](/srvdb/industries/media-journalism/fact-check-assistant/) | Matches claims against previously fact-checked claims and sources. |
| 3 · Advanced | [Narrative Tracker](/srvdb/industries/media-journalism/narrative-tracker/) | Clusters stories and posts into narratives. |
| 4 · Expert | **Investigation Workbench** (this page) | Isolated, multimodal workspace per investigation. |
| 5 · Frontier | [Secure Leak Analysis](/srvdb/industries/media-journalism/secure-leak-analysis/) | Air-gapped analysis of large leaked datasets. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
