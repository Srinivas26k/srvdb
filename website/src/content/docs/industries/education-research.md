---
title: Education & research
description: How SrvDB powers Offline Study & Paper Assistant and other local AI tools in education & research. Use cases, architecture and code pattern.
sidebar:
  order: 8
---

Students and researchers need to search papers, notes and course material by meaning, often on their own laptops.

:::note[Coming soon]
A complete, runnable **Offline Study & Paper Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Keyword search misses papers that use different terms.
- Course material is spread across files and years.
- Not every institution allows uploading material to cloud AI.

## How SrvDB helps

- **Paper search by meaning** over a personal or lab library.
- **Study assistant** that answers from course notes with citations.
- **Runs on a laptop**, offline, installed with pip.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Paper search
- Lecture-note search
- Syllabus search
- Question-bank search
- Thesis archive search
- Lab protocol search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Study assistant with citations
- Grading-rubric assistant
- Literature review assistant
- Lab protocol Q&A
- Course FAQ bot
- Proposal drafting grounded in the group's papers

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Duplicate question detection
- Text-reuse signals
- Similar-paper recommendation
- Concept gaps: student questions far from the course material
- Reviewer matching by paper similarity
- Dataset deduplication for ML research

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Personal tutor with per-student memory
- Figures and slides with multimodal embeddings
- Lab-notebook agent
- One database per course
- Offline classroom servers in low-connectivity regions
- Shared research-group knowledge base

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- On-device learning apps
- Embedding research: SrvDB as a reproducible test harness
- Reproducible retrieval benchmarks
- Cross-lingual education content linking
- Long-term memory for lifelong-learning agents
- Scientific data similarity (spectra, molecules) with domain embeddings

## Example: Offline Study & Paper Assistant

A desktop study assistant that answers questions only from the student's own notes and readings, quoting the passages it used.

**How it works:**

1. Embed papers or notes paragraph by paragraph.
2. Search by question; show passages with their source.
3. Optionally pass them to a local LLM for a summary.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

papers = srvdb.Database("./papers", dimension=768)

def related_passages(question, k=5):
    return [json.loads(papers.get(i)) for i, _ in papers.search(embed(question), k)]
```

**Recommended mode:** `flat`.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
