---
title: "Healthcare"
description: "Five SrvDB projects for healthcare, from beginner to frontier."
sidebar:
  label: Overview
  order: 0
---

Clinical text is among the most sensitive data there is, and hospitals often cannot send it to external AI services.

## The challenge

- Clinicians spend time searching notes, guidelines and discharge summaries.
- Similar past cases are hard to find by keyword alone.
- Patient data must stay inside the hospital's own systems.

## Why SrvDB fits

- **Guideline and protocol RAG:** answer questions from internal guidelines with a local LLM.
- **Similar-case search:** find de-identified past cases that resemble a current one.
- **On-premises only:** data stays in a local directory on hospital hardware.

## Five projects, five levels

Each project has its own page: the problem, how SrvDB powers it, what changes, architecture, use cases, core code and limits.

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Protocol Assistant](/srvdb/industries/healthcare/protocol-assistant/) | Answers clinicians' questions from the hospital's own protocols and formulary, with citations. |
| 2 · Intermediate | [Similar Case Finder](/srvdb/industries/healthcare/similar-case-finder/) | Finds de-identified past cases that resemble the patient in front of you. |
| 3 · Advanced | [Trial Matcher](/srvdb/industries/healthcare/trial-matcher/) | Shortlists patients for clinical trials by comparing summaries with eligibility criteria. |
| 4 · Expert | [Pharmacovigilance Monitor](/srvdb/industries/healthcare/pharmacovigilance-monitor/) | Matches new adverse-event reports to known safety signals and groups similar reports. |
| 5 · Frontier | [On-prem Clinical AI Stack](/srvdb/industries/healthcare/on-prem-clinical-ai/) | A complete hospital AI stack with zero data egress: local embeddings, local LLM, SrvDB. |

:::caution
SrvDB keeps data local, but it is not a certified medical device or a compliance product. Regulatory compliance (HIPAA, GDPR and similar) remains the deployer's responsibility.
:::

:::note[Coming soon]
Runnable reference implementations of these projects are in progress.
:::
