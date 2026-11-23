---
title: "Threat Hunter Agent"
description: "An autonomous hunting agent that reasons across network, endpoint and threat-intel memory. Level 4 Cybersecurity project powered by SrvDB."
sidebar:
  label: "L4 \u00b7 Threat Hunter Agent"
  order: 4
---

**An autonomous hunting agent that reasons across network, endpoint and threat-intel memory.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Cybersecurity](/srvdb/industries/cybersecurity/) | 4 of 5 | Expert | `flat` |

:::note[Coming soon]
A complete, runnable **Threat Hunter Agent** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Hunts are manual, occasional and depend on one analyst's time.
- Context is split across network, endpoint and intel tools.
- Results of past hunts are forgotten, so hunts are repeated.

## How SrvDB powers Threat Hunter Agent

1. Separate SrvDB databases hold network sessions, endpoint events, threat intel and the agent's own hunt memory.
2. The agent turns a hypothesis into searches, runs them in parallel (`search_batch`, thread-safe handles) and passes the evidence to a local LLM.
3. Every hunt, query and conclusion is stored in the hunt-memory database, so "have we checked this before?" is one search.
4. Stale intel is removed with `delete` and the store reclaimed with `compact()`.

## What changes

| Today | With Threat Hunter Agent on SrvDB |
|---|---|
| One hunt per analyst-day | Hypotheses are hunted continuously; analysts review findings |
| Hunt results are lost in notebooks | Every hunt is stored and recalled by meaning |
| Pivoting across tools is manual | One agent queries every database in one step |

## How it works

1. Feed network, endpoint and intel data into their own databases.
2. Give the agent a hypothesis (for example an ATT&CK technique).
3. It searches all databases in parallel and gathers evidence.
4. A local LLM evaluates the evidence and writes a finding with links.
5. The finding goes into hunt memory; escalations need human approval.

## SrvDB features it uses

- **[Several databases](/srvdb/start/use-cases/)**: each database is a directory, so different data types or tenants stay separate.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[`delete` and `compact()`](/srvdb/guides/delete-upsert-compact/)**: remove items immediately from search, then reclaim disk space.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Continuous hunting for selected ATT&CK techniques
- Pivot from a new intel report to matching sessions and endpoints
- "Have we hunted for this before, and what did we find?"
- Automatic hunt reports
- Alert prioritisation with hunt context
- Retiring stale intel with delete and compact

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./threat-hunter-agent-db", dimension=768)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def recall_hunts(hypothesis, k=5):
    hits = db.search(embed(hypothesis), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- The agent proposes; analysts decide. Every finding must link to its evidence.
- Plan databases per data type; one database can be open in one place at a time.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Cybersecurity projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [SOC Copilot](/srvdb/industries/cybersecurity/soc-copilot/) | A knowledge assistant that answers analysts' questions from your own runbooks, incidents and policies. |
| 2 · Intermediate | [Phishing Hunter](/srvdb/industries/cybersecurity/phishing-hunter/) | Catches reworded phishing by meaning, not by exact strings. |
| 3 · Advanced | [PacketSense](/srvdb/industries/cybersecurity/packetsense/) | A Wireshark-style traffic explorer where you search network sessions by behaviour. |
| 4 · Expert | **Threat Hunter Agent** (this page) | An autonomous hunting agent that reasons across network, endpoint and threat-intel memory. |
| 5 · Frontier | [Air-gapped OT Sentinel](/srvdb/industries/cybersecurity/ot-sentinel/) | Offline anomaly detection for industrial control networks, running on edge gateways. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
