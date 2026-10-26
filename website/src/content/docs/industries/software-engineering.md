---
title: Software engineering
description: How SrvDB powers Local Code Navigator and other local AI tools in software engineering. Use cases, architecture and code pattern.
sidebar:
  order: 10
---

Developers want AI help with their codebase without uploading proprietary code.

:::note[Coming soon]
A complete, runnable **Local Code Navigator** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Finding "where do we validate tokens?" by keyword is slow.
- Internal docs and code drift apart.
- Many companies forbid sending source code to external services.

## How SrvDB helps

- **Code search by meaning** with a code embedding model.
- **Keep the index current** with `upsert` and `delete` as files change.
- **Everything local**, including in CI.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Code search by meaning
- Documentation search
- Architecture-decision search
- Error-message search
- Runbook search
- Commit-history search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Codebase Q&A
- Developer onboarding
- Incident runbook assistant
- API usage assistant
- PR descriptions from similar PRs
- Test-writing assistant grounded in existing tests

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Duplicate bug reports
- Similar stack traces
- Code-clone candidates
- Flaky-test clustering
- Log anomalies
- Snippet provenance checks

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Coding-agent memory
- One database per repository
- Indexing on every commit in CI
- Offline IDE plugin
- Incident agent
- Architecture knowledge agent

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Air-gapped development environments
- Regression detection in logs by distance
- Monorepo-scale code index (benchmark first)
- Crash-safe memory for long autonomous agent runs
- Search for known vulnerable code patterns
- Self-hosted developer copilot stack

## Example: Local Code Navigator

An editor or CLI assistant that answers "where is..." and "how does..." questions about a repository using only a local index.

**How it works:**

1. Embed functions and doc sections; store path and line as metadata.
2. On each commit, upsert changed chunks and delete removed ones.
3. Search by question; open the matching locations.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

code = srvdb.Database("./code-index", dimension=768)

def where_is(question, k=5):
    return [json.loads(code.get(i))["path"] for i, _ in code.search(embed(question), k)]
```

**Recommended mode:** `flat` or `hnsw`.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
