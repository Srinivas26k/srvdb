---
title: "Scientific Similarity Lab"
description: "Similarity search over spectra, molecules and other scientific data. Level 5 Education & research project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Scientific Similarity Lab"
  order: 5
---

**Similarity search over spectra, molecules and other scientific data.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Education & research](/srvdb/industries/education-research/) | 5 of 5 | Frontier | `flat` |

:::note[Coming soon]
A complete, runnable **Scientific Similarity Lab** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Scientific data is not text.
- Comparing samples is manual.
- Results are hard to reproduce.

## How SrvDB powers Scientific Similarity Lab

1. Domain embedding models turn spectra or molecules into vectors stored in SrvDB.
2. Nearest neighbours find similar samples.
3. Exact `flat` search keeps results reproducible.
4. Rust API for pipelines.

## What changes

| Today | With Scientific Similarity Lab on SrvDB |
|---|---|
| Manual comparisons | Instant nearest samples |
| Ad-hoc scripts | A reusable store |
| Irreproducible searches | Deterministic exact search |

## How it works

1. Choose a domain embedding model.
2. Embed samples.
3. Search.
4. Validate with experts.
5. Publish the pipeline.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.

## Use cases

- Spectra matching
- Molecule similarity
- Materials screening
- Protein fragments
- Astronomy catalogues
- Embedding research

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./scientific-similarity-lab-db", dimension=256)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_samples(sample, k=5):
    hits = db.search(embed(sample), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Results depend entirely on the domain embedding model (research).

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
| 4 · Expert | [Research Group Brain](/srvdb/industries/education-research/research-group-brain/) | A shared, searchable memory for a lab: notes, data descriptions, figures. |
| 5 · Frontier | **Scientific Similarity Lab** (this page) | Similarity search over spectra, molecules and other scientific data. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
