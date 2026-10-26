---
title: "Study Assistant"
description: "Answers questions only from the student's own course material. Level 2 Education & research project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Study Assistant"
  order: 2
---

**Answers questions only from the student's own course material.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Education & research](/srvdb/industries/education-research/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Study Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Generic chatbots stray from the syllabus.
- Students need sources.
- Not all schools allow cloud AI.

## How SrvDB powers Study Assistant

1. Notes and readings are embedded locally.
2. Answers cite course passages.
3. Runs offline.
4. Per-course databases.

## What changes

| Today | With Study Assistant on SrvDB |
|---|---|
| Off-syllabus answers | Grounded answers |
| No sources | Every answer cited |
| Cloud concerns | Local |

## How it works

1. Ingest course material.
2. Ask.
3. Review cited passages.
4. Add new lectures.
5. Revise.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Exam revision
- Homework help
- Concept explanations
- Lab preparation
- Language learning
- Accessibility support

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./study-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def study_answer(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Encourage learning, not copying.

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
| 2 · Intermediate | **Study Assistant** (this page) | Answers questions only from the student's own course material. |
| 3 · Advanced | [Reviewer Matcher](/srvdb/industries/education-research/reviewer-matcher/) | Matches papers or proposals to suitable reviewers. |
| 4 · Expert | [Research Group Brain](/srvdb/industries/education-research/research-group-brain/) | A shared, searchable memory for a lab: notes, data descriptions, figures. |
| 5 · Frontier | [Scientific Similarity Lab](/srvdb/industries/education-research/scientific-similarity-lab/) | Similarity search over spectra, molecules and other scientific data. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
