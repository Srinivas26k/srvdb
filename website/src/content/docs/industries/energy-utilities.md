---
title: Energy & utilities
description: How SrvDB powers Grid Operations Assistant and other local AI tools in energy & utilities. Use cases, architecture and code pattern.
sidebar:
  order: 20
---

Utilities run grids, plants and assets with decades of documentation, often at remote sites.

:::note[Coming soon]
A complete, runnable **Grid Operations Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Procedures and asset records are scattered.
- Similar past outages hold the fastest fix.
- Field sites have limited connectivity.

## How SrvDB helps

- **Procedure and asset search** for control rooms and crews.
- **Similar-outage retrieval** from past event reports.
- **Offline on tablets and gateways**, crash-safe at remote sites.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Operating procedure search
- Asset record search
- Outage report search
- Regulatory filing search
- Safety document search
- Equipment manual search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Control-room Q&A
- Field-crew assistant
- Outage report drafting
- Regulatory Q&A
- Operator training
- Tariff questions for customer service

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Similar outage events
- SCADA event-description anomalies
- Asset failure similarity
- Meter-data pattern similarity
- Duplicate work orders
- Drone inspection image similarity

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Offline field agent on tablets
- Substation edge assistants
- Drone imagery plus inspection notes
- One database per region
- Continuous ingestion
- Storm-response knowledge agent

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Air-gapped OT-network deployments
- Forecast analogues: nearest historical weather patterns
- Crash-safe operation at remote sites
- Compact indexes on field gateways
- Pattern sharing across utilities via database snapshots
- Offline documentation assistant for non-safety plant systems

## Example: Grid Operations Assistant

A control-room and field assistant that retrieves procedures and the most similar past outages for a new event.

**How it works:**

1. Embed procedures, asset records and outage reports.
2. Search with the new event description.
3. Show the closest past events and procedures; operators decide.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

outages = srvdb.Database("./outages", dimension=384)

def similar_outages(event, k=5):
    return [json.loads(outages.get(i)) for i, _ in outages.search(embed(event), k)]
```

**Recommended mode:** `flat`; `sq8` on small gateways.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
