---
title: "PacketSense"
description: "A Wireshark-style traffic explorer where you search network sessions by behaviour. Level 3 Cybersecurity project powered by SrvDB."
sidebar:
  label: "L3 \u00b7 PacketSense"
  order: 3
---

**A Wireshark-style traffic explorer where you search network sessions by behaviour.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Cybersecurity](/srvdb/industries/cybersecurity/) | 3 of 5 | Advanced | `hnsw` |

:::note[Coming soon]
A complete, runnable **PacketSense** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Packet captures are huge, and display filters only work if you already know what to look for.
- A command-and-control beacon hides among normal-looking HTTPS sessions.
- There is no way to ask: "show me all traffic that behaves like this session".

## How SrvDB powers PacketSense

1. Zeek or tshark turn traffic into flow records; each flow becomes a vector of behaviour features (duration, bytes each way, packet-size and timing statistics, ports, protocol) and/or an embedded text summary.
2. SrvDB stores millions of these with `hnsw` for fast search, and `sq8` keeps the store 4× smaller.
3. Query by example (pick one session, find similar ones) or by description (a text query against embedded flow summaries).
4. Sessions far from every known-normal session are surfaced as anomalies.

## What changes

| Today | With PacketSense on SrvDB |
|---|---|
| Filters need exact field names and values | Point at a session or describe it, and get the similar ones |
| One beacon found means a manual hunt for the rest | One search finds every session that behaves like it |
| Anomaly baselines are hand-built | Distance from stored normal traffic does the work |

## How it works

1. Capture with tcpdump, Zeek or tshark (SrvDB does not capture packets).
2. Convert each flow into features and a short text summary.
3. Embed and store with flow id, endpoints and time as metadata.
4. Explore: query by example or by description in a notebook or small web UI.
5. Label findings and `upsert` them so later searches benefit.

## SrvDB features it uses

- **[HNSW graph (`hnsw`)](/srvdb/guides/index-modes/)**: approximate search that visits a small part of large collections; tune with `set_ef_search()`.
- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[`search_batch`](/srvdb/reference/python-api/)**: runs many queries at once, in parallel inside Rust.
- **[Thread-safe handles](/srvdb/guides/concurrency/)**: one database object can be shared between threads; searches run in parallel.
- **[`upsert`](/srvdb/guides/delete-upsert-compact/)**: inserts new items and replaces changed ones, so the knowledge base stays current.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.

## Use cases

- Find every session similar to a known C2 beacon
- Spot flows that look like DNS tunnelling
- Find lateral-movement sessions resembling a past incident
- "HTTPS sessions with long idle gaps and small regular packets"
- Per-segment baselines of normal traffic
- Triage IDS alerts with the most similar past flows
- Saved hunting queries in a notebook

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./packetsense-db", dimension=128, mode="hnsw")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def similar_sessions(session_summary, k=5):
    hits = db.search(embed(session_summary), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- SrvDB does not capture or decode packets; Zeek, tshark or similar tools do.
- Encrypted payloads leave metadata and timing only; results depend on the features you choose.
- Check `hnsw` recall against `flat` on a sample before relying on it.

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
| 3 · Advanced | **PacketSense** (this page) | A Wireshark-style traffic explorer where you search network sessions by behaviour. |
| 4 · Expert | [Threat Hunter Agent](/srvdb/industries/cybersecurity/threat-hunter-agent/) | An autonomous hunting agent that reasons across network, endpoint and threat-intel memory. |
| 5 · Frontier | [Air-gapped OT Sentinel](/srvdb/industries/cybersecurity/ot-sentinel/) | Offline anomaly detection for industrial control networks, running on edge gateways. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
