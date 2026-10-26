---
title: Anti-money laundering
description: How SrvDB powers AML Investigation Assistant and other local AI tools in anti-money laundering. Use cases, architecture and code pattern.
sidebar:
  order: 17
---

AML teams triage alerts, investigate cases, write reports and track typologies, under strict confidentiality.

:::note[Coming soon]
A complete, runnable **AML Investigation Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Most alerts are false positives that look like earlier ones.
- Report narratives are written from scratch each time.
- Typology knowledge sits with a few experienced investigators.

## How SrvDB helps

- **Alert prioritisation** by similarity to past dispositions.
- **Typology matching** of case narratives.
- **Confidential by design:** data stays on your infrastructure.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Typology search
- Internal suspicious-activity narrative search
- AML policy search
- Adverse-media archive search
- KYC document search
- Case-note search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Investigation Q&A
- Report drafting from similar past narratives (investigator reviews)
- Typology explainer
- Screening policy Q&A
- Investigator training
- Regulator guidance Q&A

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Alert prioritisation by similarity to past true and false positives
- Name and entity matching with embeddings, alongside rule-based screening
- Adverse-media deduplication and matching
- Typology matching of case narratives
- Mule-pattern similarity
- Duplicate alert consolidation

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Investigator agent with case memory
- Joint search across case, typology and adverse-media databases
- Continuous learning from dispositions
- Multilingual adverse media
- Per-jurisdiction databases
- Quality-assurance sampling agent

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Confidential on-premises stack
- Case networks linked through nearest neighbours
- Emerging typologies: clusters far from every known typology
- In-process pre-screening (benchmark first)
- Cross-border multilingual matching
- Crash-safe audit logs

## Example: AML Investigation Assistant

An investigator assistant that shows the closest past cases and their outcomes, the matching typologies, and drafts a narrative for the investigator to review.

**How it works:**

1. Embed closed case narratives with disposition as metadata, and typology descriptions.
2. For a new alert, retrieve similar cases and typologies.
3. Investigators decide; dispositions are fed back with `upsert`.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

cases = srvdb.Database("./aml-cases", dimension=384)

def similar_cases(narrative, k=5):
    return [(json.loads(cases.get(i))["disposition"], round(s, 2)) for i, s in cases.search(embed(narrative), k)]
```

**Recommended mode:** `flat` for exact, explainable scores.

:::caution
Decision support only: SrvDB does not replace regulated screening systems or investigator judgement.
:::

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
