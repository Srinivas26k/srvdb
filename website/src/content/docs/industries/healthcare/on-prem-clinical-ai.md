---
title: "On-prem Clinical AI Stack"
description: "A complete hospital AI stack with zero data egress: local embeddings, local LLM, SrvDB. Level 5 Healthcare project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 On-prem Clinical AI Stack"
  order: 5
---

**A complete hospital AI stack with zero data egress: local embeddings, local LLM, SrvDB.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Healthcare](/srvdb/industries/healthcare/) | 5 of 5 | Frontier | `hnsw` |

:::note[Coming soon]
A complete, runnable **On-prem Clinical AI Stack** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Every AI pilot raises data-protection questions.
- Each department builds its own tool.
- Rural sites lack connectivity.

## How SrvDB powers On-prem Clinical AI Stack

1. SrvDB is the retrieval layer: one database per department or data domain, all on hospital hardware.
2. Assistants share the same pattern: retrieve, answer with citations, log what was shown.
3. Crash-safe `persist()` and a local audit log of retrievals support governance.
4. Database folders can be copied to offline rural sites.

## What changes

| Today | With On-prem Clinical AI Stack on SrvDB |
|---|---|
| One-off pilots | One reusable on-prem pattern |
| Data leaves for cloud AI | No egress at all |
| Rural sites excluded | Offline copies of the knowledge base |

## How it works

1. Deploy local embedding and LLM services.
2. Create SrvDB databases per domain.
3. Expose search through internal APIs.
4. Log every retrieval for audit.
5. Sync folders to remote sites.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.
- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.

## Use cases

- Department assistants on one platform
- Audit trail of AI suggestions
- Offline rural clinic assistant
- Research sandboxes per study
- Genomics or imaging similarity research
- Ambulance tablets with protocols

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./on-prem-clinical-ai-db", dimension=768, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def retrieve(query, k=5):
    hits = db.search(embed(query), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Not a medical device; governance and compliance are the hospital's responsibility.

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
| 4 · Expert | [Pharmacovigilance Monitor](/srvdb/industries/healthcare/pharmacovigilance-monitor/) | Matches new adverse-event reports to known safety signals and groups similar reports. |
| 5 · Frontier | **On-prem Clinical AI Stack** (this page) | A complete hospital AI stack with zero data egress: local embeddings, local LLM, SrvDB. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
