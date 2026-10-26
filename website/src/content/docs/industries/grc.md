---
title: Governance, risk & compliance
description: How SrvDB powers Control Mapping Assistant and other local AI tools in governance, risk & compliance. Use cases, architecture and code pattern.
sidebar:
  order: 16
---

GRC teams map controls, policies, risks and evidence across many frameworks and regulations.

:::note[Coming soon]
A complete, runnable **Control Mapping Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- The same control is documented differently for every framework.
- Questionnaires repeat questions already answered.
- Evidence gaps surface only during the audit.

## How SrvDB helps

- **Control-mapping suggestions** by similarity between control texts (a person confirms).
- **Answer reuse** for security and vendor questionnaires.
- **Evidence gap checks:** controls with no close evidence item.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Policy search
- Control-library search
- Risk-register search
- Audit-finding search
- Regulation search
- Evidence search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Compliance Q&A
- Policy drafting grounded in existing policies
- Audit preparation assistant
- Vendor questionnaires answered from past approved answers
- Risk-assessment assistant
- Staff training

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Cross-framework control mapping suggestions (a person confirms)
- Duplicate risks
- Policy overlap and conflict detection
- Questionnaire answer reuse
- Evidence gaps: controls with no close evidence
- Issue clustering

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- GRC agent that keeps mappings and remembers reviewer feedback
- One database per legal entity
- Continuous control-monitoring ingestion
- Regulatory change impact agent
- Vendor-risk pipeline
- Board report drafting

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- On-premises GRC AI with an audit trail of retrievals
- Regulatory horizon scanning by clustering
- Cross-language regulation alignment
- Crash-safe evidence index
- Scenario analogue search
- Enterprise-wide control rationalisation

## Example: Control Mapping Assistant

A compliance assistant that proposes framework-to-framework control mappings, reuses approved questionnaire answers and lists controls lacking evidence.

**How it works:**

1. Embed control statements from each framework you use, plus policies and evidence descriptions.
2. For each control, search the other framework's controls and the evidence database.
3. Reviewers accept or reject; accepted mappings are stored with `upsert`.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

controls = srvdb.Database("./controls", dimension=768)

def mapping_candidates(control_text, k=3):
    return [(i, round(s, 2)) for i, s in controls.search(embed(control_text), k)]
```

**Recommended mode:** `flat`.

:::caution
SrvDB supports compliance work; it does not certify compliance with any framework.
:::

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
