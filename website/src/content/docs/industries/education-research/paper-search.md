---
title: "Paper Search"
description: "Search a personal or lab library of papers by meaning. Level 1 Education & research project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Paper Search"
  order: 1
---

**Search a personal or lab library of papers by meaning.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Education & research](/srvdb/industries/education-research/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Paper Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Keywords miss papers using different terms.
- Libraries grow faster than memory.
- Notes are scattered.

## How SrvDB powers Paper Search

1. Paper abstracts and paragraphs are embedded in SrvDB.
2. Questions find relevant passages.
3. Runs on a laptop.
4. New papers added with `upsert`.

## What changes

| Today | With Paper Search on SrvDB |
|---|---|
| Keyword search | Concept search |
| Forgotten papers | Rediscovered by meaning |
| Scattered notes | One searchable library |

## How it works

1. Extract text.
2. Embed paragraphs.
3. Search.
4. Save findings.
5. Add new papers.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Literature search
- Related-work drafting
- Reading lists
- Citation hunting
- Thesis research
- Grant writing

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./paper-search-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def related_passages(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- PDF extraction quality matters.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Education & research projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Paper Search** (this page) | Search a personal or lab library of papers by meaning. |
| 2 · Intermediate | [Study Assistant](/srvdb/industries/education-research/study-assistant/) | Answers questions only from the student's own course material. |
| 3 · Advanced | [Reviewer Matcher](/srvdb/industries/education-research/reviewer-matcher/) | Matches papers or proposals to suitable reviewers. |
| 4 · Expert | [Research Group Brain](/srvdb/industries/education-research/research-group-brain/) | A shared, searchable memory for a lab: notes, data descriptions, figures. |
| 5 · Frontier | [Scientific Similarity Lab](/srvdb/industries/education-research/scientific-similarity-lab/) | Similarity search over spectra, molecules and other scientific data. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
