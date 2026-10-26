---
title: Fiscal auditing
description: How SrvDB powers Audit Working-Paper Assistant and other local AI tools in fiscal auditing. Use cases, architecture and code pattern.
sidebar:
  order: 18
---

Internal, external and public-sector auditors review ledgers, contracts, invoices and prior working papers.

:::note[Coming soon]
A complete, runnable **Audit Working-Paper Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Prior-year findings and working papers are hard to reuse.
- Duplicate invoices and vendors hide behind different wording.
- Unusual journal entries are buried in volume.

## How SrvDB helps

- **Working-paper and finding search** across engagements.
- **Duplicate candidates** for invoices and vendors by description similarity.
- **Unusual-entry flags:** descriptions far from typical ones.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Working-paper search
- Auditing-standards search
- Prior-findings search
- Contract search
- Invoice description search
- Policy search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Audit-programme Q&A
- Finding drafting grounded in past findings
- Standards Q&A
- Walkthrough documentation assistant
- Management-letter drafting
- Staff training

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Duplicate invoice candidates
- Unusual journal-entry descriptions far from normal ones
- Vendor-master duplicates
- Similar findings across entities
- Expense-claim anomalies
- Contract-clause deviations

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Audit agent with engagement memory
- Per-engagement databases to support independence
- Continuous-auditing pipeline
- Receipt images with multimodal embeddings
- Public-sector budget review agent
- Sampling assistant

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Offline field audits
- Cross-entity pattern discovery
- Procurement fraud signals for review
- Crash-safe evidence index
- Multilingual audits
- Multi-year archive search

## Example: Audit Working-Paper Assistant

An auditor assistant that retrieves relevant prior findings and standards passages and lists duplicate and unusual items for follow-up.

**How it works:**

1. Embed prior findings, standards text you hold, invoice and journal descriptions.
2. Search per audit area; run duplicate and outlier checks on the period's items.
3. Auditors review every flagged item.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

invoices = srvdb.Database("./invoices", dimension=384)

def duplicate_candidates(description, threshold=0.95):
    return [i for i, s in invoices.search(embed(description), k=5) if s >= threshold]
```

**Recommended mode:** `flat` for exact scores.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
