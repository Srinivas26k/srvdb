---
title: "Trial Matcher"
description: "Shortlists patients for clinical trials by comparing summaries with eligibility criteria. Level 3 Healthcare project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Trial Matcher"
  order: 3
---

**Shortlists patients for clinical trials by comparing summaries with eligibility criteria.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Healthcare](/srvdb/industries/healthcare/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Trial Matcher** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Screening for trials is slow and manual.
- Eligible patients are missed.
- Criteria are long free text.

## How SrvDB powers Trial Matcher

1. Each trial's inclusion criteria are embedded; patient summaries are compared against them.
2. Top matches form a shortlist for coordinators.
3. `search_batch` screens many patients at once.
4. Closed trials are removed with `delete`.

## What changes

| Today | With Trial Matcher on SrvDB |
|---|---|
| Coordinators read every chart | A ranked shortlist comes first |
| Patients missed for trials | Every summary is checked against every open trial |
| Spreadsheets of criteria | Criteria are searchable by meaning |

## How it works

1. Embed open trials' criteria.
2. Build de-identified patient summaries.
3. Run `search_batch` across summaries.
4. Coordinators review the shortlist.
5. Update as trials open and close.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Oncology trial pre-screening
- Rare-disease recruitment
- Investigator dashboards
- Feasibility estimates for new trials
- Referral suggestions between sites
- Re-screening when criteria change

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./trial-matcher-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def matching_trials(patient_summary, k=5):
    hits = db.search(embed(patient_summary), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- A shortlist for humans, never an eligibility decision.
- Structured criteria (labs, ages) still need rule checks.

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
| 3 · Advanced | **Trial Matcher** (this page) | Shortlists patients for clinical trials by comparing summaries with eligibility criteria. |
| 4 · Expert | [Pharmacovigilance Monitor](/srvdb/industries/healthcare/pharmacovigilance-monitor/) | Matches new adverse-event reports to known safety signals and groups similar reports. |
| 5 · Frontier | [On-prem Clinical AI Stack](/srvdb/industries/healthcare/on-prem-clinical-ai/) | A complete hospital AI stack with zero data egress: local embeddings, local LLM, SrvDB. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
