---
title: "Narrative Tracker"
description: "Clusters stories and posts into narratives. Level 3 Media & journalism project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Narrative Tracker"
  order: 3
---

**Clusters stories and posts into narratives.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Media & journalism](/srvdb/industries/media-journalism/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Narrative Tracker** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Narratives spread across outlets.
- Duplicates obscure origins.
- Trends emerge fast.

## How SrvDB powers Narrative Tracker

1. Items are embedded and grouped by nearest neighbours.
2. New clusters signal emerging narratives.
3. Duplicates collapse.
4. `search_batch` for feeds.

## What changes

| Today | With Narrative Tracker on SrvDB |
|---|---|
| Manual monitoring | Automatic clusters |
| Duplicates | Collapsed |
| Late trend detection | Early clusters |

## How it works

1. Ingest feeds.
2. Embed.
3. Cluster.
4. Review.
5. Report.

## SrvDB features it uses

- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Wire duplicate removal
- Disinformation narratives
- Topic monitoring
- Quote attribution
- Plagiarism checks
- Audience trends

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./narrative-tracker-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def narrative_neighbours(item, k=5):
    hits = db.search(embed(item), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Analysts interpret clusters.

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
| 3 · Advanced | **Narrative Tracker** (this page) | Clusters stories and posts into narratives. |
| 4 · Expert | [Investigation Workbench](/srvdb/industries/media-journalism/investigation-workbench/) | Isolated, multimodal workspace per investigation. |
| 5 · Frontier | [Secure Leak Analysis](/srvdb/industries/media-journalism/secure-leak-analysis/) | Air-gapped analysis of large leaked datasets. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
