---
title: Government & defence
description: How SrvDB powers Air-gapped Knowledge Assistant and other local AI tools in government & defence. Use cases, architecture and code pattern.
sidebar:
  order: 5
---

Classified and isolated networks cannot use cloud AI at all.

:::note[Coming soon]
A complete, runnable **Air-gapped Knowledge Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Doctrine, manuals and reports are scattered and hard to search.
- Systems must run with no internet access, ever.
- Every dependency has to be reviewed.

## How SrvDB helps

- **Fully offline:** no network access, no telemetry, no background service.
- **Small, reviewable footprint:** one library, no Python runtime dependencies, Apache-2.0, a published SBOM with every release.
- **Crash-safe:** committed data survives power loss on field hardware.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Manual and doctrine search
- Policy archive search
- Procurement document search
- Information-request archive search
- Translation-memory search
- Equipment part lookup by description

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Air-gapped doctrine Q&A
- Citizen-service assistant over public guidance
- Procurement compliance assistant
- Briefing drafts from retrieved reports
- Training tutor
- Legislative history Q&A

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Duplicate request detection
- Report linking by similarity
- Maintenance-log anomalies
- Narrative clustering of public disinformation content
- Bid similarity signals in procurement (for human review)
- Open-source document deduplication

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Field assistant on rugged laptops, fully offline
- Separation by classification: one database per level
- Image and text retrieval with multimodal embeddings
- Continuous ingestion with `persist()` checkpoints
- Translation assistant with memory
- Mission-planning knowledge agent

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Fully air-gapped AI stack with an SBOM-reviewed dependency chain
- Disconnected edge deployments on vehicles and ships
- Cross-lingual intelligence linking
- Crash-safe operation on unreliable power
- Builds from reviewed source for supply-chain assurance
- Long-term archival memory with versioned on-disk formats

## Example: Air-gapped Knowledge Assistant

A question-answering assistant over manuals and reports that runs on a single isolated machine, with every answer linked to its source passages.

**How it works:**

1. Install from a wheel copied into the isolated network.
2. Embed documents with a local model; store source and section as metadata.
3. Answer questions with a local LLM, always showing sources.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

library = srvdb.Database("./library", dimension=768)

def cite(question, k=5):
    return [json.loads(library.get(i))["source"] for i, _ in library.search(embed(question), k)]
```

**Recommended mode:** `flat` or `hnsw` depending on size.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
