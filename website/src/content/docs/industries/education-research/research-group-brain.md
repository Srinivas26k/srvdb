---
title: "Research Group Brain"
description: "A shared, searchable memory for a lab: notes, data descriptions, figures. Level 4 Education & research project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Research Group Brain"
  order: 4
---

**A shared, searchable memory for a lab: notes, data descriptions, figures.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Education & research](/srvdb/industries/education-research/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Research Group Brain** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Knowledge leaves with graduating students.
- Lab notebooks are unsearchable.
- Figures are hard to find.

## How SrvDB powers Research Group Brain

1. Notes, protocols and figure captions (and images with multimodal embeddings) go into SrvDB.
2. Everyone searches one memory.
3. An agent drafts summaries.
4. Crash-safe storage on a lab server.

## What changes

| Today | With Research Group Brain on SrvDB |
|---|---|
| Knowledge leaves with people | Knowledge stays |
| Unsearchable notebooks | Searchable memory |
| Lost figures | Found by description |

## How it works

1. Collect notes and figures.
2. Embed.
3. Search and summarise.
4. Add continuously.
5. Back up the folder.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.

## Use cases

- Onboarding students
- Protocol reuse
- Figure retrieval
- Grant reports
- Reproducibility
- Collaboration

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./research-group-brain-db", dimension=512)

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

- Curate what goes in.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Education & research projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Paper Search](/srvdb/industries/education-research/paper-search/) | Search a personal or lab library of papers by meaning. |
| 2 · Intermediate | [Study Assistant](/srvdb/industries/education-research/study-assistant/) | Answers questions only from the student's own course material. |
| 3 · Advanced | [Reviewer Matcher](/srvdb/industries/education-research/reviewer-matcher/) | Matches papers or proposals to suitable reviewers. |
| 4 · Expert | **Research Group Brain** (this page) | A shared, searchable memory for a lab: notes, data descriptions, figures. |
| 5 · Frontier | [Scientific Similarity Lab](/srvdb/industries/education-research/scientific-similarity-lab/) | Similarity search over spectra, molecules and other scientific data. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
