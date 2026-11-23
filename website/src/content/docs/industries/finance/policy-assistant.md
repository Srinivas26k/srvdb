---
title: "Policy Assistant"
description: "Answers staff questions from internal policies and product terms. Level 1 Finance & banking project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 Policy Assistant"
  order: 1
---

**Answers staff questions from internal policies and product terms.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Finance & banking](/srvdb/industries/finance/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **Policy Assistant** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Policies are long and change often.
- Front-line staff need quick, correct answers.
- Customer data must stay internal.

## How SrvDB powers Policy Assistant

1. Policies and product terms are embedded in SrvDB.
2. Questions retrieve passages; a local LLM answers with citations.
3. Updates via `upsert`.
4. Runs inside the bank's network.

## What changes

| Today | With Policy Assistant on SrvDB |
|---|---|
| Email the compliance team | Instant answers with sources |
| Outdated PDFs | Current passages only |
| Cloud AI blocked | Local stack |

## How it works

1. Collect policies.
2. Embed.
3. Answer with citations.
4. Review usage.
5. Refresh on change.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- Lending policy questions
- Fee and product terms
- KYC procedure questions
- Complaint handling steps
- Data-handling rules
- Branch procedures

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./policy-assistant-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def policy_answer(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Answers must show the source policy.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Finance & banking projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **Policy Assistant** (this page) | Answers staff questions from internal policies and product terms. |
| 2 · Intermediate | [Complaint Analyzer](/srvdb/industries/finance/complaint-analyzer/) | Clusters and deduplicates customer complaints to find root causes. |
| 3 · Advanced | [Fraud Pattern Matcher](/srvdb/industries/finance/fraud-pattern-matcher/) | Compares new cases with confirmed fraud cases and ranks them for investigators. |
| 4 · Expert | [Investigator Agent](/srvdb/industries/finance/investigator-agent/) | An agent that gathers context across cases, policies and customer history. |
| 5 · Frontier | [On-prem Risk AI Platform](/srvdb/industries/finance/on-prem-risk-ai/) | A data-resident AI retrieval platform for risk and compliance teams. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
