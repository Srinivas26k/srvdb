---
title: Quantum computing
description: How SrvDB powers Quantum Lab Notebook Assistant and other local AI tools in quantum computing. Use cases, architecture and code pattern.
sidebar:
  order: 19
---

Quantum teams produce papers, circuits, calibration data and experiment logs, much of it proprietary research.

:::note[Coming soon]
A complete, runnable **Quantum Lab Notebook Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Results from earlier runs are hard to find again.
- Calibration drift is noticed late.
- Group knowledge is scattered across notebooks and chats.

## How SrvDB helps

- **Lab-notebook search and Q&A** over the group's own material.
- **Calibration and run similarity:** compare snapshots stored as vectors.
- **Local and private** for unpublished research.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Paper search
- Experiment-log search
- Calibration procedure search
- Circuit library search by description
- Hardware note and error-code search
- Search over Qiskit, Cirq or other SDK code snippets

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Lab Q&A over the group's own notes
- Literature review assistant
- Onboarding tutor
- Experiment write-up drafting
- Hardware runbook assistant
- Algorithm explainer grounded in group notes

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Calibration drift: compare calibration snapshots as vectors
- Similar past experiments
- Run-log anomalies
- Equivalent-circuit candidates via circuit embeddings (research)
- Noise-profile similarity across devices
- Paper clustering

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Lab agent remembering runs
- Plots plus logs with multimodal embeddings
- One database per device
- Pipeline indexing every run
- Memory for hybrid classical-quantum workflows
- Compiler-pass knowledge agent

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Circuit embeddings for similarity search (research)
- Error-syndrome pattern retrieval (research)
- Nearest previously computed molecules in quantum chemistry
- Classical result store for quantum ML experiments
- Reproducibility archive of experiments
- Air-gapped lab knowledge base for proprietary research

## Example: Quantum Lab Notebook Assistant

A lab assistant that answers questions from the group's notes and finds past runs and calibrations most similar to the current one.

**How it works:**

1. Embed notes and run summaries; store calibration snapshots as feature vectors.
2. Search with the current run's summary or calibration vector.
3. Review the closest past runs and what was done about them.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

runs = srvdb.Database("./runs", dimension=384)

def similar_runs(summary, k=5):
    return [(i, round(s, 2)) for i, s in runs.search(embed(summary), k)]
```

**Recommended mode:** `flat`.

:::caution
SrvDB is classical software: it stores ordinary vectors describing experiments and runs on regular computers, not on quantum hardware.
:::

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
