---
title: Legal
description: How SrvDB powers Clause & Precedent Finder and other local AI tools in legal. Use cases, architecture and code pattern.
sidebar:
  order: 3
---

Law firms and legal teams hold privileged documents that must never leave their control.

:::note[Coming soon]
A complete, runnable **Clause & Precedent Finder** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Finding a clause like the one in front of you means reading hundreds of contracts.
- Relevant precedent is phrased differently from the question.
- Client confidentiality rules out many cloud tools.

## How SrvDB helps

- **Clause similarity:** embed every clause; find the closest wording across all past contracts.
- **Precedent retrieval:** search case summaries by meaning.
- **Privilege stays local:** the index is a folder on the firm's machines.

## Example: Clause & Precedent Finder

A drafting companion that, for any clause being written, shows the closest clauses the firm has used before, with the contract and date they came from.

**How it works:**

1. Split contracts into clauses; store contract id, clause type and date as metadata.
2. While drafting, search with the current clause to see precedents.
3. Use `delete` when documents must be removed under a retention policy, then `compact()`.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

clauses = srvdb.Database("./clauses", dimension=768)

def closest_clauses(draft, k=5):
    return [(json.loads(clauses.get(i)), round(s, 3)) for i, s in clauses.search(embed(draft), k)]
```

**Recommended mode:** `flat`; `sq8` to save disk on very large archives.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
