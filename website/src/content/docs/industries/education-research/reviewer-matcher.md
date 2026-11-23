---
title: "Reviewer Matcher"
description: "Matches papers or proposals to suitable reviewers. Level 3 Education & research project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Reviewer Matcher"
  order: 3
---

**Matches papers or proposals to suitable reviewers.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Education & research](/srvdb/industries/education-research/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Reviewer Matcher** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Finding reviewers is slow.
- Conflicts and mismatches happen.
- Expertise is described in prose.

## How SrvDB powers Reviewer Matcher

1. Reviewers' publications are embedded.
2. Submissions query the closest experts.
3. Scores rank candidates.
4. Pools updated with `upsert`.

## What changes

| Today | With Reviewer Matcher on SrvDB |
|---|---|
| Manual reviewer search | Ranked candidates |
| Mismatched reviews | Better matches |
| Static pools | Updated pools |

## How it works

1. Embed reviewer profiles.
2. Embed submissions.
3. Rank matches.
4. Check conflicts.
5. Assign.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Journal reviews
- Conference committees
- Grant panels
- Thesis examiners
- Internal reviews
- Mentor matching

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./reviewer-matcher-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def find_reviewers(submission_abstract, k=5):
    hits = db.search(embed(submission_abstract), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Conflict-of-interest checks remain manual.

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
| 3 · Advanced | **Reviewer Matcher** (this page) | Matches papers or proposals to suitable reviewers. |
| 4 · Expert | [Research Group Brain](/srvdb/industries/education-research/research-group-brain/) | A shared, searchable memory for a lab: notes, data descriptions, figures. |
| 5 · Frontier | [Scientific Similarity Lab](/srvdb/industries/education-research/scientific-similarity-lab/) | Similarity search over spectra, molecules and other scientific data. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
