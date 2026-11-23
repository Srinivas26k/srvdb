---
title: "Similar Case Finder"
description: "Finds de-identified past cases that resemble the patient in front of you. Level 2 Healthcare project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Similar Case Finder"
  order: 2
---

**Finds de-identified past cases that resemble the patient in front of you.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Healthcare](/srvdb/industries/healthcare/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Similar Case Finder** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Rare presentations are hard to compare without a colleague who has seen one.
- Case archives are searchable only by codes.
- Teaching files are scattered.

## How SrvDB powers Similar Case Finder

1. De-identified case summaries are embedded and stored in SrvDB.
2. A summary of the current case is used as the query; the closest cases come back with their outcomes.
3. Cosine scores show how close each case is.
4. New cases are added with `upsert` after review.

## What changes

| Today | With Similar Case Finder on SrvDB |
|---|---|
| Code-based search misses similar stories | Search by the clinical narrative itself |
| Knowledge depends on who is on duty | The whole archive is available to everyone |
| Teaching files are hard to curate | Similar cases cluster naturally |

## How it works

1. De-identify case summaries.
2. Embed and store with diagnosis and outcome metadata.
3. Query with the current case summary.
4. Show matches and outcomes for clinician review.
5. Add reviewed cases regularly.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Rare-disease differentials
- Teaching-case retrieval
- Morbidity and mortality review preparation
- Similar imaging-report narratives
- Research cohort scoping
- Second-opinion preparation

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./similar-case-finder-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_cases(case_summary, k=5):
    hits = db.search(embed(case_summary), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- De-identification must happen before ingestion.
- Similar is not diagnostic; clinicians interpret.

:::caution
SrvDB keeps data local, but it is not a certified medical device or a compliance product. Regulatory compliance (HIPAA, GDPR and similar) remains the deployer's responsibility.
:::

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Healthcare projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Protocol Assistant](/srvdb/industries/healthcare/protocol-assistant/) | Answers clinicians' questions from the hospital's own protocols and formulary, with citations. |
| 2 · Intermediate | **Similar Case Finder** (this page) | Finds de-identified past cases that resemble the patient in front of you. |
| 3 · Advanced | [Trial Matcher](/srvdb/industries/healthcare/trial-matcher/) | Shortlists patients for clinical trials by comparing summaries with eligibility criteria. |
| 4 · Expert | [Pharmacovigilance Monitor](/srvdb/industries/healthcare/pharmacovigilance-monitor/) | Matches new adverse-event reports to known safety signals and groups similar reports. |
| 5 · Frontier | [On-prem Clinical AI Stack](/srvdb/industries/healthcare/on-prem-clinical-ai/) | A complete hospital AI stack with zero data egress: local embeddings, local LLM, SrvDB. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
