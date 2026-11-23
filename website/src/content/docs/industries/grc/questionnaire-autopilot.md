---
title: "Questionnaire Autopilot"
description: "Answers security and vendor questionnaires from past approved answers. Level 2 Governance, risk & compliance project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Questionnaire Autopilot"
  order: 2
---

**Answers security and vendor questionnaires from past approved answers.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Governance, risk & compliance](/srvdb/industries/grc/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Questionnaire Autopilot** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Questionnaires repeat the same questions.
- Answers are scattered.
- Responses take weeks.

## How SrvDB powers Questionnaire Autopilot

1. Approved answers are embedded.
2. Each question retrieves the closest answers.
3. A local LLM drafts responses with sources.
4. Approved responses fed back.

## What changes

| Today | With Questionnaire Autopilot on SrvDB |
|---|---|
| Weeks of manual work | Drafts in minutes |
| Inconsistent answers | Approved answers reused |
| Lost knowledge | Growing library |

## How it works

1. Embed approved answers.
2. Import questionnaire.
3. Draft answers.
4. Review.
5. Save approved.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Security questionnaires
- Vendor due diligence
- RFP responses
- Customer audits
- Privacy assessments
- Insurance forms

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./questionnaire-autopilot-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def best_answers(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Owners approve every answer.

:::caution
SrvDB supports compliance work; it does not certify compliance with any framework.
:::

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Governance, risk & compliance projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Policy Search](/srvdb/industries/grc/policy-search/) | Find the right policy passage from a plain question. |
| 2 · Intermediate | **Questionnaire Autopilot** (this page) | Answers security and vendor questionnaires from past approved answers. |
| 3 · Advanced | [Control Mapper](/srvdb/industries/grc/control-mapper/) | Suggests mappings between controls across frameworks. |
| 4 · Expert | [Regulatory Change Agent](/srvdb/industries/grc/regulatory-change-agent/) | Assesses which policies and controls a new regulation affects. |
| 5 · Frontier | [Continuous Compliance Platform](/srvdb/industries/grc/continuous-compliance-platform/) | On-prem AI retrieval for evidence, controls and audits with an audit trail. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
