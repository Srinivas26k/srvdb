---
title: Finance & banking
description: How SrvDB powers Fraud Pattern Matcher and other local AI tools in finance & banking. Use cases, architecture and code pattern.
sidebar:
  order: 4
---

Financial institutions need similarity search over transactions, cases and policies, under strict data-residency rules.

:::note[Coming soon]
A complete, runnable **Fraud Pattern Matcher** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- New fraud resembles old fraud, but not exactly.
- Analysts search policy manuals and regulatory text by keyword.
- Customer records are duplicated across systems.

## How SrvDB helps

- **Case similarity:** embed descriptions of confirmed fraud cases; compare new cases against them.
- **Policy RAG:** answer compliance questions from internal manuals.
- **Deduplication:** flag near-duplicate customer or case records with a cosine threshold.
- **Data residency:** everything stays in a local directory you control.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Policy manual search
- Product terms search
- Analyst research note search
- Past credit memo search
- Customer complaint search
- Regulatory circular search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Compliance Q&A with citations
- Credit memo drafting from the closest past memos
- Customer-service assistant over product terms
- Q&A over filings and reports you hold
- Internal-audit finding explainer
- Earnings-call notes Q&A

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Fraud case similarity to confirmed cases
- Complaint clustering and deduplication
- KYC record deduplication
- Unusual transaction narratives far from normal ones
- Mule-account behaviour similarity
- Trade-surveillance messages similar to known misconduct examples (for review)

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Investigator agent with case memory
- Joint search across case, policy and typology databases
- Continuously updated fraud patterns via `upsert`
- Offline assistants in branches
- Per-client advisor memory
- Document intake with multimodal embeddings (statements, IDs)

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- On-premises AI stack meeting data-residency rules
- In-process pre-screening inside payment flows (benchmark first)
- Synthetic-identity ring discovery through nearest-neighbour graphs
- Model-risk documentation retrieval for validators
- Historical market-narrative analogues for stress scenarios
- Crash-safe decision logs

## Example: Fraud Pattern Matcher

An investigator's assistant that compares a new case narrative with confirmed fraud cases and lists the closest matches with their similarity scores, for a human to review.

**How it works:**

1. Embed confirmed case narratives with case id and outcome as metadata.
2. Score each new case against them; route high-similarity cases to investigators.
3. Add confirmed outcomes with `upsert` so matching improves over time.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

cases = srvdb.Database("./fraud-cases", dimension=384)

def resembles_known_fraud(narrative, threshold=0.85):
    hits = cases.search(embed(narrative), k=3)
    return [(i, round(s, 2)) for i, s in hits if s >= threshold]
```

**Recommended mode:** `flat` for exact scores; thresholds mean the same in every mode.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
