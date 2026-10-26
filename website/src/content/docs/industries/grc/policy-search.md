---
title: "Policy Search"
description: "Find the right policy passage from a plain question. Level 1 Governance, risk & compliance project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Policy Search"
  order: 1
---

**Find the right policy passage from a plain question.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Governance, risk & compliance](/srvdb/industries/grc/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Policy Search** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Policies are spread over many documents.
- Staff can't find the rule.
- Versions drift.

## How SrvDB powers Policy Search

1. Policies embedded by section.
2. Questions retrieve passages.
3. Updates via `upsert`.
4. Local deployment.

## What changes

| Today | With Policy Search on SrvDB |
|---|---|
| Searching folders | Instant passages |
| Outdated versions | Current only |
| Ad-hoc answers | Sourced answers |

## How it works

1. Collect policies.
2. Embed.
3. Search.
4. Track questions.
5. Update.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Acceptable use
- Data classification
- Access control
- Retention
- Incident reporting
- Third-party rules

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./policy-search-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def policy_passages(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Policy owners keep documents current.

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
| 1 · Beginner | **Policy Search** (this page) | Find the right policy passage from a plain question. |
| 2 · Intermediate | [Questionnaire Autopilot](/srvdb/industries/grc/questionnaire-autopilot/) | Answers security and vendor questionnaires from past approved answers. |
| 3 · Advanced | [Control Mapper](/srvdb/industries/grc/control-mapper/) | Suggests mappings between controls across frameworks. |
| 4 · Expert | [Regulatory Change Agent](/srvdb/industries/grc/regulatory-change-agent/) | Assesses which policies and controls a new regulation affects. |
| 5 · Frontier | [Continuous Compliance Platform](/srvdb/industries/grc/continuous-compliance-platform/) | On-prem AI retrieval for evidence, controls and audits with an audit trail. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
