---
title: "Control Mapper"
description: "Suggests mappings between controls across frameworks. Level 3 Governance, risk & compliance project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 Control Mapper"
  order: 3
---

**Suggests mappings between controls across frameworks.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Governance, risk & compliance](/srvdb/industries/grc/) | 3 of 5 | Advanced | `flat` |

:::note[Coming soon]
A complete, runnable **Control Mapper** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Each framework words controls differently.
- Mapping is manual and repeated.
- Evidence gets duplicated.

## How SrvDB powers Control Mapper

1. Control statements from each framework are embedded.
2. Each control retrieves its closest counterparts.
3. Reviewers accept or reject.
4. Accepted mappings stored.

## What changes

| Today | With Control Mapper on SrvDB |
|---|---|
| Spreadsheet mapping | Suggested mappings |
| Duplicate evidence | Shared evidence |
| Repeated effort | Reusable mappings |

## How it works

1. Embed controls.
2. Suggest mappings.
3. Review.
4. Store accepted.
5. Link evidence.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Framework crosswalks
- Evidence reuse
- Gap analysis
- Audit planning
- Policy alignment
- Mergers

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./control-mapper-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def mapping_candidates(control_text, k=5):
    hits = db.search(embed(control_text), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Suggestions only; reviewers decide.

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
| 2 · Intermediate | [Questionnaire Autopilot](/srvdb/industries/grc/questionnaire-autopilot/) | Answers security and vendor questionnaires from past approved answers. |
| 3 · Advanced | **Control Mapper** (this page) | Suggests mappings between controls across frameworks. |
| 4 · Expert | [Regulatory Change Agent](/srvdb/industries/grc/regulatory-change-agent/) | Assesses which policies and controls a new regulation affects. |
| 5 · Frontier | [Continuous Compliance Platform](/srvdb/industries/grc/continuous-compliance-platform/) | On-prem AI retrieval for evidence, controls and audits with an audit trail. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
