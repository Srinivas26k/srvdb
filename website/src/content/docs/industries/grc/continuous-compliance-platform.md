---
title: "Continuous Compliance Platform"
description: "On-prem AI retrieval for evidence, controls and audits with an audit trail. Level 5 Governance, risk & compliance project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Continuous Compliance Platform"
  order: 5
---

**On-prem AI retrieval for evidence, controls and audits with an audit trail.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Governance, risk & compliance](/srvdb/industries/grc/) | 5 of 5 | Frontier | `hnsw` |

:::note[Coming soon]
A complete, runnable **Continuous Compliance Platform** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Compliance is periodic and manual.
- Evidence lives everywhere.
- Auditors need traceability.

## How SrvDB powers Continuous Compliance Platform

1. Databases for controls, evidence and findings.
2. Continuous ingestion with `upsert`.
3. Every retrieval logged.
4. Crash-safe evidence index.

## What changes

| Today | With Continuous Compliance Platform on SrvDB |
|---|---|
| Annual scrambles | Continuous readiness |
| Scattered evidence | Indexed evidence |
| No traceability | Logged retrievals |

## How it works

1. Connect evidence sources.
2. Index continuously.
3. Check gaps.
4. Report.
5. Audit.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.
- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.

## Use cases

- Evidence gap checks
- Audit preparation
- Control monitoring
- Issue clustering
- Board dashboards
- Multi-entity groups

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./continuous-compliance-platform-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def evidence_for(control, k=5):
    hits = db.search(embed(control), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Does not certify compliance.

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
| 4 · Expert | [Regulatory Change Agent](/srvdb/industries/grc/regulatory-change-agent/) | Assesses which policies and controls a new regulation affects. |
| 5 · Frontier | **Continuous Compliance Platform** (this page) | On-prem AI retrieval for evidence, controls and audits with an audit trail. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
