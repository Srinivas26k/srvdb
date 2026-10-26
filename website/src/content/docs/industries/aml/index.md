---
title: "Anti-money laundering"
description: "Five SrvDB projects for anti-money laundering, from beginner to frontier."
sidebar:
  label: Overview
  order: 0
---

AML teams triage alerts, investigate cases, write reports and track typologies, under strict confidentiality.

## The challenge

- Most alerts are false positives that look like earlier ones.
- Report narratives are written from scratch each time.
- Typology knowledge sits with a few experienced investigators.

## Why SrvDB fits

- **Alert prioritisation** by similarity to past dispositions.
- **Typology matching** of case narratives.
- **Confidential by design:** data stays on your infrastructure.

## Five projects, five levels

Each project has its own page: the problem, how SrvDB powers it, what changes, architecture, use cases, core code and limits.

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Typology Library](/srvdb/industries/aml/typology-library/) | Search money-laundering typologies and red flags by description. |
| 2 · Intermediate | [Alert Prioritizer](/srvdb/industries/aml/alert-prioritizer/) | Ranks AML alerts by similarity to past true and false positives. |
| 3 · Advanced | [Adverse Media Matcher](/srvdb/industries/aml/adverse-media-matcher/) | Deduplicates and matches adverse media to customers. |
| 4 · Expert | [AML Investigation Agent](/srvdb/industries/aml/aml-investigation-agent/) | An investigator agent with memory across cases, typologies and media. |
| 5 · Frontier | [Emerging Typology Radar](/srvdb/industries/aml/emerging-typology-radar/) | Detects clusters of cases far from every known typology. |

:::caution
Decision support only: SrvDB does not replace regulated screening systems or investigator judgement.
:::

:::note[Coming soon]
Runnable reference implementations of these projects are in progress.
:::
