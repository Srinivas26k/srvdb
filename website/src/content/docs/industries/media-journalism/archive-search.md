---
title: "Archive Search"
description: "Search decades of articles by meaning. Level 1 Media & journalism project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Archive Search"
  order: 1
---

**Search decades of articles by meaning.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Media & journalism](/srvdb/industries/media-journalism/) | 1 of 5 | Beginner | `hnsw` |

:::note[Coming soon]
A complete, runnable **Archive Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Archive search needs exact keywords.
- Old coverage is rediscovered by accident.
- Background research is slow.

## How SrvDB powers Archive Search

1. Articles are embedded paragraph by paragraph.
2. Queries find relevant coverage with date and headline.
3. `hnsw` for large archives.
4. New stories added daily.

## What changes

| Today | With Archive Search on SrvDB |
|---|---|
| Keyword archives | Meaning-based search |
| Missed context | Earlier coverage found |
| Slow research | Instant background |

## How it works

1. Embed archive.
2. Search.
3. Cite sources.
4. Add daily.
5. Curate.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Background research
- Anniversary stories
- Fact context
- Obituaries
- Explainers
- Data journalism

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./archive-search-db", dimension=384, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def earlier_coverage(text, k=5):
    hits = db.search(embed(text), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Archive digitisation quality matters.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Media & journalism projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Archive Search** (this page) | Search decades of articles by meaning. |
| 2 · Intermediate | [Fact-check Assistant](/srvdb/industries/media-journalism/fact-check-assistant/) | Matches claims against previously fact-checked claims and sources. |
| 3 · Advanced | [Narrative Tracker](/srvdb/industries/media-journalism/narrative-tracker/) | Clusters stories and posts into narratives. |
| 4 · Expert | [Investigation Workbench](/srvdb/industries/media-journalism/investigation-workbench/) | Isolated, multimodal workspace per investigation. |
| 5 · Frontier | [Secure Leak Analysis](/srvdb/industries/media-journalism/secure-leak-analysis/) | Air-gapped analysis of large leaked datasets. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
