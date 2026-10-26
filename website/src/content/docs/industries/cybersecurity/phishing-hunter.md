---
title: "Phishing Hunter"
description: "Catches reworded phishing by meaning, not by exact strings. Level 2 Cybersecurity project powered by SrvDB."
sidebar:
  label: "L2 \u00b7 Phishing Hunter"
  order: 2
---

**Catches reworded phishing by meaning, not by exact strings.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Cybersecurity](/srvdb/industries/cybersecurity/) | 2 of 5 | Intermediate | `flat` |

:::note[Coming soon]
A complete, runnable **Phishing Hunter** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Campaigns change a few words and slip past exact-match rules.
- User-reported emails flood the queue, many of them the same campaign.
- Business-email-compromise lures have no malicious link to detect.

## How SrvDB powers Phishing Hunter

1. Confirmed phishing emails (subject, body, sender display name) are embedded as a reference set in SrvDB.
2. Every new or user-reported email is scored by cosine similarity against that set; scores mean the same thing in every mode, so one threshold works.
3. Emails above the threshold go to quarantine review; near-identical reports are grouped automatically.
4. Analyst verdicts are added with `upsert`, so the reference set learns every new campaign.

## What changes

| Today | With Phishing Hunter on SrvDB |
|---|---|
| One changed word breaks a rule | A reworded lure still scores high because the meaning is the same |
| Each user report is triaged by hand | Reports matching a known campaign are grouped and pre-labelled |
| Campaigns are invisible as a whole | Nearest-neighbour groups show each campaign and its variants |

## How it works

1. Collect confirmed phishing and benign samples.
2. Embed subject, body and sender display name.
3. Store each with its campaign label as metadata.
4. Score incoming mail: top-3 neighbours and their scores.
5. Above threshold: quarantine for review; the analyst's verdict is `upsert`ed back.

## SrvDB features it uses

- **[Exact search (`flat`)](/srvdb/guides/index-modes/)**: every query compares against every stored vector, so results are exact and explainable.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Detect reworded variants of a known campaign
- Group user-reported emails by campaign
- Flag CEO-fraud requests similar to past business-email-compromise cases
- Spot brand-impersonation lures
- Collapse duplicate reports
- Campaign reports for management
- Find real examples of a lure type for awareness training

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./phishing-hunter-db", dimension=384)

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def score_email(email_text, k=5):
    hits = db.search(embed(email_text), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Complements your email gateway; it does not replace it.
- Needs a labelled set of real phishing examples to start.

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
| 2 · Intermediate | **Phishing Hunter** (this page) | Catches reworded phishing by meaning, not by exact strings. |
| 3 · Advanced | [PacketSense](/srvdb/industries/cybersecurity/packetsense/) | A Wireshark-style traffic explorer where you search network sessions by behaviour. |
| 4 · Expert | [Threat Hunter Agent](/srvdb/industries/cybersecurity/threat-hunter-agent/) | An autonomous hunting agent that reasons across network, endpoint and threat-intel memory. |
| 5 · Frontier | [Air-gapped OT Sentinel](/srvdb/industries/cybersecurity/ot-sentinel/) | Offline anomaly detection for industrial control networks, running on edge gateways. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
