---
title: "Air-gapped OT Sentinel"
description: "Offline anomaly detection for industrial control networks, running on edge gateways. Level 5 Cybersecurity project powered by SrvDB."
sidebar:
  label: "L5 \u00b7 Air-gapped OT Sentinel"
  order: 5
---

**Offline anomaly detection for industrial control networks, running on edge gateways.**

| Industry | Level | Difficulty | Recommended mode |
|---|---|---|---|
| [Cybersecurity](/srvdb/industries/cybersecurity/) | 5 of 5 | Frontier | `sq8` |

:::note[Coming soon]
A complete, runnable **Air-gapped OT Sentinel** reference implementation is in progress. The architecture and core code below already work with SrvDB 0.3.
:::

## The problem

- Industrial (OT) networks are isolated and cannot use cloud analytics.
- Control devices are fragile, so monitoring must be passive.
- Remote sites lose power, and local state must survive it.

## How SrvDB powers Air-gapped OT Sentinel

1. A baseline of normal control traffic and command sequences is embedded as feature windows and stored with `sq8` on the gateway.
2. Each new window is compared with the baseline; a large distance raises a local alert.
3. `persist()` makes the baseline crash-safe: after a power cut the gateway reopens exactly as of the last commit.
4. The Rust API embeds SrvDB in a small gateway service; SrvDB itself never opens a network connection.

## What changes

| Today | With Air-gapped OT Sentinel on SrvDB |
|---|---|
| Cloud analytics are not allowed on OT networks | Everything runs on the site gateway |
| Signature IDS misses new behaviour | Deviation from learned normal is flagged |
| Power loss can corrupt local state | The baseline reopens intact after a crash |

## How it works

1. Passive network tap feeding an ICS-aware parser (for example Zeek with ICS protocol support).
2. Build fixed-length feature windows of commands and timings (at least 128 dimensions, SrvDB's minimum).
3. Store baseline windows in SrvDB (`sq8`) inside a Rust gateway service.
4. Score live windows; alert on the local HMI when the distance is high.
5. Engineers review alerts and update the baseline during maintenance windows.

## SrvDB features it uses

- **[SQ8 compression (`sq8`)](/srvdb/guides/index-modes/)**: stores each value in one byte, 4× smaller than full precision.
- **[Rust API](/srvdb/guides/rust/)**: embed SrvDB directly in Rust services and devices.
- **[Crash-safe `persist()`](/srvdb/guides/durability/)**: after it returns, data survives a crash or power loss and reopens exactly as committed.
- **[Fully offline](/srvdb/start/introduction/)**: SrvDB runs in-process and never opens a network connection.
- **[True cosine scores](/srvdb/guides/index-modes/)**: scores in [-1, 1] mean the same in every mode, so thresholds are stable.

## Use cases

- Unusual Modbus or DNP3 command sequences
- New behaviour from a known device
- Engineering-workstation activity outside its normal pattern
- Firmware-update traffic outside maintenance windows
- Comparing sites by shipping baseline snapshots
- Forensic replay of anomalous windows

## Core code

`embed()` is your embedding model (for example sentence-transformers, or a function that turns your records into feature vectors).

```python
import json
import srvdb

db = srvdb.Database("./ot-sentinel-db", dimension=128, mode="sq8")

def ingest(items):
    """items: list of (id, text, metadata dict)"""
    db.upsert(
        ids=[i for i, _, _ in items],
        embeddings=[embed(t) for _, t, _ in items],
        metadatas=[json.dumps(m) for _, _, m in items],
    )
    db.persist()                      # crash-safe checkpoint

def deviation(window_summary, k=5):
    hits = db.search(embed(window_summary), k)
    return [(json.loads(db.get(i)), round(score, 3)) for i, score in hits]
```

## Limits and honest notes

- Needs a careful baseline and threshold tuning per site.
- Not a certified safety system; it monitors passively and alerts people.

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
| 4 · Expert | [Threat Hunter Agent](/srvdb/industries/cybersecurity/threat-hunter-agent/) | An autonomous hunting agent that reasons across network, endpoint and threat-intel memory. |
| 5 · Frontier | **Air-gapped OT Sentinel** (this page) | Offline anomaly detection for industrial control networks, running on edge gateways. |

See also: [Use cases](/srvdb/start/use-cases/) · [Live examples](/srvdb/examples/overview/) (coming soon)
