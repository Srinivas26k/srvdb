---
title: Manufacturing & IoT
description: How SrvDB powers Factory-floor Maintenance Assistant and other local AI tools in manufacturing & iot. Use cases, architecture and code pattern.
sidebar:
  order: 6
---

Factories and field sites need answers at the machine, often on edge hardware with poor connectivity.

:::note[Coming soon]
A complete, runnable **Factory-floor Maintenance Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Technicians search thick manuals for the right procedure.
- Past fault reports that match today's symptoms are hard to find.
- Edge devices have little memory and unreliable networks.

## How SrvDB helps

- **Manual and fault-report search** on the device itself.
- **Small storage:** SQ8 stores vectors 4× smaller, PQ up to 32× smaller.
- **Linux aarch64 wheels** for ARM edge hardware; works with no connectivity.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Machine manual search
- Fault-code lookup
- SOP search
- Spare-part lookup by description
- Quality report search
- Supplier specification search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Maintenance assistant
- Operator training tutor
- Root-cause analysis grounded in past 8D reports
- Changeover procedure Q&A
- Health-and-safety Q&A
- Supplier quality Q&A

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Similar-fault retrieval
- Defect image similarity
- Duplicate work orders
- Sensor-window anomaly by distance
- Supplier non-conformance clustering
- Production-log anomalies

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Technician agent with shift memory
- One database per production line
- Vision plus text retrieval
- Edge gateways receiving database snapshots
- Learning from resolved work orders
- Knowledge layer for a digital twin

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Embedded memory in Rust services next to the machine
- Predictive maintenance: nearest neighbours of pre-failure signatures
- Fleet-wide pattern sharing by shipping database snapshots
- Crash-safe operation through power cuts
- Compressed indexes on small ARM Linux devices
- Golden-batch comparison of process-parameter embeddings

## Example: Factory-floor Maintenance Assistant

A tablet or edge-box assistant where a technician describes a symptom and gets the closest past fault reports and the relevant manual pages.

**How it works:**

1. Embed manuals and fault reports on a workstation, then copy the database folder to the device.
2. On the device, search by symptom description.
3. Add new fault reports locally and `persist()`.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

faults = srvdb.Database("./faults", dimension=384, mode="sq8")

def similar_faults(symptom, k=5):
    return [json.loads(faults.get(i)) for i, _ in faults.search(embed(symptom), k)]
```

**Recommended mode:** `sq8` on small devices; check recall against `flat` first.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
