---
title: "SOC Copilot"
description: "A knowledge assistant that answers analysts' questions from your own runbooks, incidents and policies. Level 1 Cybersecurity project powered by SrvDB."
sidebar:
  label: "L1 \u00b7 SOC Copilot"
  order: 1
---

**A knowledge assistant that answers analysts' questions from your own runbooks, incidents and policies.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Cybersecurity](/srvdb/industries/cybersecurity/) | 1 of 5 | Beginner | `flat` |

:::note[Coming soon]
A complete, runnable **SOC Copilot** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Analysts search wikis and ticket systems under time pressure while an alert is live.
- How a similar incident was handled last year lives in a senior analyst's head.
- New analysts take months to learn the playbooks, and cloud AI is not allowed for incident data.

## How SrvDB powers SOC Copilot

1. Every runbook section, closed incident and policy paragraph is embedded and stored in SrvDB on the SOC's own hardware.
2. Each analyst question becomes one `search`; the top passages go to a local LLM, which answers **only** from them and cites each source.
3. Closed incidents are added with `upsert` every night, so the copilot knows yesterday's cases today.
4. `persist()` after each ingestion makes the knowledge base crash-safe through reboots and power loss.

## What changes

| Today | With SOC Copilot on SrvDB |
|---|---|
| Finding the right runbook takes minutes of wiki searching | The right steps appear from a plain-language description of the alert |
| Past incident knowledge depends on who is on shift | Every closed case is searchable by everyone, instantly |
| Cloud AI is off-limits for incident data | Everything runs on SOC machines; nothing leaves the network |

## How it works

1. Export runbooks, playbooks, policies and closed tickets.
2. Split them into passages and embed each with a local model.
3. Store passages in SrvDB with source document and ticket id as metadata.
4. On a question: `search` for the 5 closest passages, then have a local LLM answer with citations.
5. Nightly: `upsert` newly closed tickets and call `persist()`.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.

## Use cases

- "What is our playbook for ransomware on a file server?"
- Explain an unfamiliar alert type from internal documentation
- "How did we handle a similar incident last year?"
- Search previous shift-handover notes
- Onboarding Q&A for new analysts
- Policy questions: data handling, escalation contacts, legal hold
- Draft an incident summary from the most similar past reports

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./soc-copilot-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def ask_soc(question, k=5):
    hits = db.search(embed(question), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Answers are only as good as your documents; keep runbooks current.
- LLMs can be wrong: always show the retrieved sources next to the answer.

## Build roadmap

1. **Prototype:** load a sample of real data with `flat` mode and check that searches return what an expert expects.
2. **Measure:** if you move to an approximate mode, compare its results with `flat` on the same queries (recall).
3. **Pipeline:** add continuous `upsert` and decide when to call `persist()`.
4. **Harden:** test restarts and crashes, add logging, review every output path with domain experts.
5. **Deploy:** ship the database directory with the application or service.

## All Cybersecurity projects

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | **SOC Copilot** (this page) | A knowledge assistant that answers analysts' questions from your own runbooks, incidents and policies. |
| 2 · Intermediate | [Phishing Hunter](/srvdb/industries/cybersecurity/phishing-hunter/) | Catches reworded phishing by meaning, not by exact strings. |
| 3 · Advanced | [PacketSense](/srvdb/industries/cybersecurity/packetsense/) | A Wireshark-style traffic explorer where you search network sessions by behaviour. |
| 4 · Expert | [Threat Hunter Agent](/srvdb/industries/cybersecurity/threat-hunter-agent/) | An autonomous hunting agent that reasons across network, endpoint and threat-intel memory. |
| 5 · Frontier | [Air-gapped OT Sentinel](/srvdb/industries/cybersecurity/ot-sentinel/) | Offline anomaly detection for industrial control networks, running on edge gateways. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
