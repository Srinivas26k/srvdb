---
title: "Protocol Assistant"
description: "Answers clinicians' questions from the hospital's own protocols and formulary, with citations. Level 1 Healthcare project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Protocol Assistant"
  order: 1
---

**Answers clinicians' questions from the hospital's own protocols and formulary, with citations.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Healthcare](/srvdb/industries/healthcare/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Protocol Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Clinicians hunt through intranet PDFs for the current protocol.
- Local formulary rules differ from general references.
- Clinical text cannot be sent to external AI services.

## How SrvDB powers Protocol Assistant

1. Protocols, formulary entries and local policies are split into passages, embedded and stored in SrvDB on hospital servers.
2. Each question is one `search`; a local LLM answers only from the retrieved passages and quotes them.
3. Protocol updates replace old passages with `upsert`; withdrawn documents are removed with `delete`.
4. `persist()` keeps the index consistent through restarts.

## What changes

| Today | With Protocol Assistant on SrvDB |
|---|---|
| Searching PDFs on the intranet | A plain-language question returns the exact protocol passage |
| Out-of-date copies circulate | Updates replace old passages the same day |
| External AI is not permitted | Everything runs on hospital infrastructure |

## How it works

1. Collect approved protocols, formulary and policies.
2. Chunk and embed with a local model.
3. Store with document, version and section as metadata.
4. Answer questions with citations via a local LLM.
5. Re-ingest on every document release.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Dosing guidance from the local formulary
- Sepsis or stroke pathway steps
- Infection-control rules
- Pre-operative checklists
- Ward-specific procedures
- Escalation and on-call policies

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./protocol-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def find_protocol(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Decision support only; clinicians remain responsible.
- Only approved, current documents should be ingested.

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
| 1 · Beginner | **Protocol Assistant** (this page) | Answers clinicians' questions from the hospital's own protocols and formulary, with citations. |
| 2 · Intermediate | [Similar Case Finder](/srvdb/industries/healthcare/similar-case-finder/) | Finds de-identified past cases that resemble the patient in front of you. |
| 3 · Advanced | [Trial Matcher](/srvdb/industries/healthcare/trial-matcher/) | Shortlists patients for clinical trials by comparing summaries with eligibility criteria. |
| 4 · Expert | [Pharmacovigilance Monitor](/srvdb/industries/healthcare/pharmacovigilance-monitor/) | Matches new adverse-event reports to known safety signals and groups similar reports. |
| 5 · Frontier | [On-prem Clinical AI Stack](/srvdb/industries/healthcare/on-prem-clinical-ai/) | A complete hospital AI stack with zero data egress: local embeddings, local LLM, SrvDB. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
