---
title: "Pharmacovigilance Monitor"
description: "Matches new adverse-event reports to known safety signals and groups similar reports. Level 4 Healthcare project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Pharmacovigilance Monitor"
  order: 4
---

**Matches new adverse-event reports to known safety signals and groups similar reports.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Healthcare](/srvdb/industries/healthcare/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Pharmacovigilance Monitor** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Adverse-event reports arrive as free text in volume.
- Duplicates inflate counts.
- Emerging signals are noticed late.

## How SrvDB powers Pharmacovigilance Monitor

1. Known signals and past reports are embedded in separate databases.
2. Each new report is matched to the closest signals and to similar reports.
3. Reports far from every known signal are flagged for review.
4. Continuous ingestion with `upsert` and periodic `persist()`.

## What changes

| Today | With Pharmacovigilance Monitor on SrvDB |
|---|---|
| Manual reading of every report | Reports arrive pre-grouped by similarity |
| Duplicates counted twice | Near-duplicates are linked |
| Signals emerge slowly | Unmatched clusters surface early |

## How it works

1. Ingest reports from your intake system.
2. Embed narratives.
3. Match to signal and report databases.
4. Queue unmatched or dense clusters for safety review.
5. Record reviewer decisions.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Duplicate report detection
- Signal matching
- Emerging-cluster review
- Literature case matching
- Periodic safety report preparation
- Product-specific dashboards

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./pharmacovigilance-monitor-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def match_report(report_text, k=5):
    hits = db.search(embed(report_text), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Supports, never replaces, regulated pharmacovigilance processes.

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
| 2 · Intermediate | [Similar Case Finder](/srvdb/industries/healthcare/similar-case-finder/) | Finds de-identified past cases that resemble the patient in front of you. |
| 3 · Advanced | [Trial Matcher](/srvdb/industries/healthcare/trial-matcher/) | Shortlists patients for clinical trials by comparing summaries with eligibility criteria. |
| 4 · Expert | **Pharmacovigilance Monitor** (this page) | Matches new adverse-event reports to known safety signals and groups similar reports. |
| 5 · Frontier | [On-prem Clinical AI Stack](/srvdb/industries/healthcare/on-prem-clinical-ai/) | A complete hospital AI stack with zero data egress: local embeddings, local LLM, SrvDB. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
