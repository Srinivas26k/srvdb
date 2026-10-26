---
title: Aerospace
description: How SrvDB powers Maintenance & Safety Knowledge Assistant and other local AI tools in aerospace. Use cases, architecture and code pattern.
sidebar:
  order: 15
---

Aerospace runs on manuals, maintenance records, safety reports and certification evidence, much of it export-controlled.

:::note[Coming soon]
A complete, runnable **Maintenance & Safety Knowledge Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Technicians search large manual sets under time pressure.
- Similar past findings across a fleet are hard to connect.
- Export-controlled data cannot go to cloud services.

## How SrvDB helps

- **Manual and bulletin search** with citations to the exact task.
- **Similar-finding retrieval** across maintenance records.
- **Offline and local:** suitable for controlled environments.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Maintenance manual search
- Service bulletin search
- Part lookup
- Incident report search
- Certification document search
- Procedure search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Technician Q&A with citations
- Safety report Q&A
- Certification evidence assistant
- Training tutor
- Flight-operations manual Q&A
- Supplier quality Q&A

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Similar maintenance findings
- Safety report clustering
- Anomalies in maintenance narratives
- Duplicate defect reports
- Similar flight-data events from embedded windows
- Non-conformance similarity

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Offline line-maintenance agent on tablets
- Fleet sharing through database snapshots
- Borescope images plus text
- Per-programme databases for export-controlled data
- Continuous ingestion
- Ground-station assistant

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Air-gapped stacks suited to export-controlled programmes
- Onboard knowledge assistant for non-flight-critical use
- Satellite and ground-segment anomaly retrieval
- Crash-safe operation
- Predictive maintenance from pre-failure signatures
- Offline mission knowledge for space missions

## Example: Maintenance & Safety Knowledge Assistant

A line-maintenance assistant on an offline tablet that retrieves the relevant manual tasks and similar past findings for a reported defect.

**How it works:**

1. Embed manuals, service bulletins and de-identified findings.
2. Search with the defect description from the technician.
3. Show tasks and past findings with their references; the technician decides.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

findings = srvdb.Database("./findings", dimension=384)

def similar_findings(defect, k=5):
    return [json.loads(findings.get(i)) for i, _ in findings.search(embed(defect), k)]
```

**Recommended mode:** `flat` or `hnsw`.

:::caution
SrvDB is not certified software. Use it for decision support and knowledge retrieval, never in flight-critical or certified systems.
:::

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
