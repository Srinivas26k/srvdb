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
