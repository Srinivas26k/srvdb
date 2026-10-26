---
title: Media & journalism
description: How SrvDB powers Archive Researcher and other local AI tools in media & journalism. Use cases, architecture and code pattern.
sidebar:
  order: 11
---

Newsrooms sit on decades of archives and sensitive source material.

:::note[Coming soon]
A complete, runnable **Archive Researcher** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Finding earlier coverage of a story means guessing the right keywords.
- Leaked or source documents must not go to third-party services.
- Fact-checking needs the original passage, fast.

## How SrvDB helps

- **Archive search by meaning** across years of articles.
- **Source protection:** sensitive document sets are indexed on an offline machine.
- **Passage-level results** with the article and date as metadata.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Archive search
- Transcript search
- Photo caption search
- Style-guide search
- Source-note search
- Wire-copy search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Research assistant
- Fact-check assistant
- Background briefings
- Style Q&A
- Interview preparation
- Translation memory

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Duplicate wire stories
- Narrative clustering
- Quote attribution matching
- Plagiarism detection
- Similar-image detection
- Claim matching against fact-checked claims

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Newsroom agent
- Video frames plus transcripts
- Per-investigation isolated databases
- Live ingestion
- Podcast search
- Audience Q&A bot

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Offline secure investigations
- Air-gapped analysis of leaked document sets
- Cross-lingual story linking
- Archive digitisation pipeline
- Provenance tracking
- Crash-safe use on field laptops

## Example: Archive Researcher

A research assistant that finds earlier coverage and the exact passages related to a new story, on the newsroom's own hardware.

**How it works:**

1. Embed articles paragraph by paragraph with headline and date.
2. Search with a draft paragraph or question.
3. Keep sensitive collections in a separate database on an offline machine.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

archive = srvdb.Database("./archive", dimension=384)

def earlier_coverage(text, k=10):
    return [json.loads(archive.get(i)) for i, _ in archive.search(embed(text), k)]
```

**Recommended mode:** `hnsw` for large archives.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
