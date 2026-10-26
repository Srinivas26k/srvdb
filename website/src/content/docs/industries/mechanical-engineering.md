---
title: Mechanical engineering
description: How SrvDB powers Design Knowledge Assistant and other local AI tools in mechanical engineering. Use cases, architecture and code pattern.
sidebar:
  order: 13
---

Engineering knowledge lives in drawings, standards, test reports and design reviews collected over decades.

:::note[Coming soon]
A complete, runnable **Design Knowledge Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Teams redesign parts that already exist under another number.
- Lessons from past failure analyses are forgotten.
- Standards and test reports are hard to search by meaning.

## How SrvDB helps

- **Part and design search** by functional description.
- **Failure-analysis retrieval:** find past investigations that match a new failure.
- **Runs on engineering workstations**, offline, with no server to manage.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Part search by description
- Material datasheet search
- Search over licensed design standards
- Test report search
- CAD metadata search
- Tolerance guideline lookup

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Design-review assistant
- FMEA assistant grounded in past FMEAs
- Standards Q&A
- Design-for-manufacturing Q&A
- Material selection assistant
- Root-cause report drafting

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Near-duplicate parts to cut part proliferation
- Similar failure modes
- Vibration-signature similarity
- Test-result anomalies
- Supplier drawing similarity
- Similar simulation-result summaries

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Engineering agent with project memory
- Geometric search with 3D shape embeddings
- Drawings plus text with multimodal embeddings
- PLM integration pipeline
- Per-programme isolated databases
- Field-failure feedback loop

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Nearest existing designs for generative-design candidates
- Condition monitoring embedded on machines
- Simulation lookup: nearest previously simulated cases
- Knowledge retention from retiring experts
- Offline shop-floor tablets
- Long-lived archives with versioned on-disk formats

## Example: Design Knowledge Assistant

An engineering assistant that, for a new design or failure, retrieves the closest existing parts, FMEAs and test reports, and answers questions from them with citations.

**How it works:**

1. Embed part descriptions, FMEAs, test reports and design-review notes.
2. Search with the new requirement or failure description.
3. Review the closest matches before designing or concluding.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

designs = srvdb.Database("./designs", dimension=768)

def existing_parts_like(requirement, k=10):
    return [(i, round(s, 2)) for i, s in designs.search(embed(requirement), k)]
```

**Recommended mode:** `flat`; `hnsw` for very large part libraries.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
