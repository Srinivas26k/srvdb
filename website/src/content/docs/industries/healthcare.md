---
title: Healthcare
description: How SrvDB powers Clinical Notes Assistant and other local AI tools in healthcare. Use cases, architecture and code pattern.
sidebar:
  order: 2
---

Clinical text is among the most sensitive data there is, and hospitals often cannot send it to external AI services.

:::note[Coming soon]
A complete, runnable **Clinical Notes Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Clinicians spend time searching notes, guidelines and discharge summaries.
- Similar past cases are hard to find by keyword alone.
- Patient data must stay inside the hospital's own systems.

## How SrvDB helps

- **Guideline and protocol RAG:** answer questions from internal guidelines with a local LLM.
- **Similar-case search:** find de-identified past cases that resemble a current one.
- **On-premises only:** data stays in a local directory on hospital hardware.

## Example: Clinical Notes Assistant

A ward-side assistant that answers questions from the hospital's own protocols and formularies, citing the passages it used, running entirely on hospital hardware.

**How it works:**

1. Embed protocols, formularies and (de-identified) case summaries.
2. For each question, retrieve the closest passages and show them with their source.
3. A local LLM drafts an answer from those passages only; the clinician decides.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

protocols = srvdb.Database("./protocols", dimension=384)

def sources_for(question, k=4):
    return [json.loads(protocols.get(i)) for i, _ in protocols.search(embed(question), k)]
```

**Recommended mode:** `flat`.

:::caution
SrvDB keeps data local, but it is not a certified medical device or a compliance product. Regulatory compliance (HIPAA, GDPR and similar) remains the deployer's responsibility.
:::

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
