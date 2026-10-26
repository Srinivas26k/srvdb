---
title: Electrical & electronics
description: How SrvDB powers Hardware Design Assistant and other local AI tools in electrical & electronics. Use cases, architecture and code pattern.
sidebar:
  order: 14
---

Electrical and electronics teams work from datasheets, schematics, test logs and compliance reports.

:::note[Coming soon]
A complete, runnable **Hardware Design Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- The answer is on page 140 of a datasheet nobody remembers.
- Component shortages need fast, reliable substitutes.
- Bench and test logs pile up without being searchable.

## How SrvDB helps

- **Datasheet and app-note Q&A** grounded in your own document library.
- **Substitute candidates** by similarity of specification text (an engineer verifies).
- **Test-log search and anomaly flags** by distance from normal logs.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Component datasheet search
- Errata search
- Application-note search
- Test-log search
- Standards search
- BOM part lookup by description

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Datasheet Q&A
- Design-rule assistant
- EMC test-failure Q&A
- Hardware/firmware interface Q&A
- Board bring-up checklist assistant
- Protection-system manual Q&A

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Second-source candidates by specification similarity
- Fault-waveform similarity
- Duplicate defect reports
- PCB defect image similarity
- Test-log anomalies
- Grid disturbance event similarity

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Lab assistant remembering bench sessions
- Oscilloscope captures plus notes
- One database per project
- Hardware-in-the-loop CI indexing results
- Shortage agent finding substitutes
- Field-return analysis agent

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- On-device anomaly detection on embedded Linux
- Relay event retrieval at substations, offline
- RF spectrum-fingerprint similarity (research)
- Crash-safe logging in harsh environments
- Air-gapped chip-design knowledge base to protect IP
- Analog circuit topology similarity (research)

## Example: Hardware Design Assistant

A bench-side assistant that answers datasheet questions with page citations, finds similar past test failures and proposes substitute parts for an engineer to check.

**How it works:**

1. Embed datasheets, app notes and test logs; store part number and page as metadata.
2. Ask questions or search with a failing test's log excerpt.
3. Engineers verify every suggestion against the source page.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

datasheets = srvdb.Database("./datasheets", dimension=768)

def datasheet_pages(question, k=5):
    return [json.loads(datasheets.get(i)) for i, _ in datasheets.search(embed(question), k)]
```

**Recommended mode:** `flat`.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
