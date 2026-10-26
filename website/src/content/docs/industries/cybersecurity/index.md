---
title: "Cybersecurity"
description: "Five SrvDB projects for cybersecurity, from beginner to frontier."
sidebar:
  label: Overview
  order: 0
---

Security teams drown in alerts, logs and threat reports, and much of that data is too sensitive to send to a cloud AI service.

## The challenge

- Analysts re-investigate incidents that look like ones already solved.
- Phishing campaigns reuse wording with small changes that defeat exact-match rules.
- Threat intelligence lives in hundreds of reports nobody can search by meaning.
- Many SOCs run on isolated networks where cloud AI is not allowed.

## Why SrvDB fits

- **Similar-incident search:** embed alert descriptions and investigation notes; a new alert instantly shows the closest past cases and how they were resolved.
- **Phishing similarity:** embed known phishing emails; a new message that scores above a cosine threshold against them is flagged, even if reworded.
- **Threat-intel RAG:** ask questions over reports and advisories with a local LLM.
- **Air-gapped by design:** SrvDB never opens a network connection and never logs vector or metadata contents.

## Five projects, five levels

Each project has its own page: the problem, how SrvDB powers it, what changes, architecture, use cases, core code and limits.

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [SOC Copilot](/srvdb/industries/cybersecurity/soc-copilot/) | A knowledge assistant that answers analysts' questions from your own runbooks, incidents and policies. |
| 2 · Intermediate | [Phishing Hunter](/srvdb/industries/cybersecurity/phishing-hunter/) | Catches reworded phishing by meaning, not by exact strings. |
| 3 · Advanced | [PacketSense](/srvdb/industries/cybersecurity/packetsense/) | A Wireshark-style traffic explorer where you search network sessions by behaviour. |
| 4 · Expert | [Threat Hunter Agent](/srvdb/industries/cybersecurity/threat-hunter-agent/) | An autonomous hunting agent that reasons across network, endpoint and threat-intel memory. |
| 5 · Frontier | [Air-gapped OT Sentinel](/srvdb/industries/cybersecurity/ot-sentinel/) | Offline anomaly detection for industrial control networks, running on edge gateways. |

:::note[Coming soon]
Runnable reference implementations of these projects are in progress.
:::
