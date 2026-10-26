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
