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
