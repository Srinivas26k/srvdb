---
title: "Fact-check Assistant"
description: "Matches claims against previously fact-checked claims and sources. Level 2 Media & journalism project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Fact-check Assistant"
  order: 2
---

**Matches claims against previously fact-checked claims and sources.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Media & journalism](/srvdb/industries/media-journalism/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Fact-check Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- The same false claims recirculate.
- Checking takes time.
- Sources are scattered.

## How SrvDB powers Fact-check Assistant

1. Fact-checked claims and verdicts are embedded.
2. New claims retrieve matches with verdicts.
3. Sources cited.
4. Updated continuously.

## What changes

| Today | With Fact-check Assistant on SrvDB |
|---|---|
| Re-checking old claims | Instant matches |
| Scattered sources | Linked sources |
| Slow response | Faster checks |

## How it works

1. Embed checked claims.
2. Query with new claims.
3. Review matches.
4. Publish.
5. Add verdicts.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Political claims
- Health misinformation
- Viral posts
- Statistics checks
- Quote verification
- Election coverage

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./fact-check-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def matching_claims(claim, k=5):
    hits = db.search(embed(claim), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Editors make the final call.

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
| 2 · Intermediate | **Fact-check Assistant** (this page) | Matches claims against previously fact-checked claims and sources. |
| 3 · Advanced | [Narrative Tracker](/srvdb/industries/media-journalism/narrative-tracker/) | Clusters stories and posts into narratives. |
| 4 · Expert | [Investigation Workbench](/srvdb/industries/media-journalism/investigation-workbench/) | Isolated, multimodal workspace per investigation. |
| 5 · Frontier | [Secure Leak Analysis](/srvdb/industries/media-journalism/secure-leak-analysis/) | Air-gapped analysis of large leaked datasets. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
