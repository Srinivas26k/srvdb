---
title: "Finance & banking"
description: "Five SrvDB projects for finance & banking, from beginner to frontier."
sidebar:
  label: Overview
  order: 0
---

Financial institutions need similarity search over transactions, cases and policies, under strict data-residency rules.

## The challenge

- New fraud resembles old fraud, but not exactly.
- Analysts search policy manuals and regulatory text by keyword.
- Customer records are duplicated across systems.

## Why SrvDB fits

- **Case similarity:** embed descriptions of confirmed fraud cases; compare new cases against them.
- **Policy RAG:** answer compliance questions from internal manuals.
- **Deduplication:** flag near-duplicate customer or case records with a cosine threshold.
- **Data residency:** everything stays in a local directory you control.

## Five projects, five levels

Each project has its own page: the problem, how SrvDB powers it, what changes, architecture, use cases, core code and limits.

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Policy Assistant](/srvdb/industries/finance/policy-assistant/) | Answers staff questions from internal policies and product terms. |
| 2 · Intermediate | [Complaint Analyzer](/srvdb/industries/finance/complaint-analyzer/) | Clusters and deduplicates customer complaints to find root causes. |
| 3 · Advanced | [Fraud Pattern Matcher](/srvdb/industries/finance/fraud-pattern-matcher/) | Compares new cases with confirmed fraud cases and ranks them for investigators. |
| 4 · Expert | [Investigator Agent](/srvdb/industries/finance/investigator-agent/) | An agent that gathers context across cases, policies and customer history. |
| 5 · Frontier | [On-prem Risk AI Platform](/srvdb/industries/finance/on-prem-risk-ai/) | A data-resident AI retrieval platform for risk and compliance teams. |

:::note[Coming soon]
Runnable reference implementations of these projects are in progress.
:::
