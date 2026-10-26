---
title: "Regulatory Change Agent"
description: "Assesses which policies and controls a new regulation affects. Level 4 Governance, risk & compliance project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Regulatory Change Agent"
  order: 4
---

**Assesses which policies and controls a new regulation affects.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Governance, risk & compliance](/srvdb/industries/grc/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Regulatory Change Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Regulations change constantly.
- Impact analysis is slow.
- Ownership is unclear.

## How SrvDB powers Regulatory Change Agent

1. New regulatory text is matched against policies and controls databases.
2. The agent drafts an impact summary.
3. Every link cites sources.
4. Decisions remembered.

## What changes

| Today | With Regulatory Change Agent on SrvDB |
|---|---|
| Manual impact analysis | Drafted impact |
| Unclear ownership | Linked owners |
| Forgotten decisions | Remembered |

## How it works

1. Ingest regulation.
2. Search policy and control databases.
3. Draft impact.
4. Assign owners.
5. Record decisions.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.

## Use cases

- New data-protection rules
- Sector regulations
- Standards updates
- Internal policy changes
- Board reporting
- Horizon scanning

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./regulatory-change-agent-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def affected_items(regulation_text, k=5):
    hits = db.search(embed(regulation_text), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Legal review required.

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
| 3 · Advanced | [Control Mapper](/srvdb/industries/grc/control-mapper/) | Suggests mappings between controls across frameworks. |
| 4 · Expert | **Regulatory Change Agent** (this page) | Assesses which policies and controls a new regulation affects. |
| 5 · Frontier | [Continuous Compliance Platform](/srvdb/industries/grc/continuous-compliance-platform/) | On-prem AI retrieval for evidence, controls and audits with an audit trail. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
