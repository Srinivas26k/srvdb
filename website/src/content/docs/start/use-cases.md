---
title: Use cases
description: What SrvDB is good for, with examples, the right mode for each, and when to choose something else.
---

SrvDB fits anywhere an application needs **similarity search over
embeddings, inside its own process, on local disk**. Each use case below
says why SrvDB fits and which [index mode](/srvdb/guides/index-modes/) to
start with.

## 1. Local RAG over private documents

Answer questions from PDFs, notes, wikis or tickets with a local LLM, without
sending documents anywhere. Chunks are embedded, stored with their text as
metadata, and the closest chunks become the LLM's context.

- **Why SrvDB:** no server to run, data stays on the machine, `upsert` makes
  re-indexing changed files simple, `persist()` makes the index crash-safe.
- **Start with:** `flat`; `hnsw` for very large corpora.
- **Guide:** [Build a local RAG app](/srvdb/guides/local-rag/).

## 2. Long-term memory for AI agents

Give an assistant memory across sessions: store facts, past messages and
tool results as embeddings, recall the relevant ones each turn, forget with
`delete`.

```python
memory = srvdb.Database("./agent-memory", dimension=384)

def remember(key, text):
    memory.upsert([key], [embed(text)], [json.dumps({"text": text})])
    memory.persist()

def recall(query, k=5):
    return [json.loads(memory.get(i))["text"] for i, _ in memory.search(embed(query), k)]
```

- **Why SrvDB:** in-process (no latency to a service), safe across crashes,
  one directory per user or per agent.
- **Start with:** `flat`.

## 3. Semantic search in desktop and CLI apps

"Search by meaning" in a notes app, email client, file manager or terminal
tool, shipped as part of the app.

- **Why SrvDB:** `pip install srvdb` with no runtime dependencies, wheels for
  Linux, macOS and Windows; nothing for users to install or configure.
- **Start with:** `flat`, or `auto` to switch to HNSW as the data grows.

## 4. Offline and air-gapped systems

Healthcare, legal, finance, defence and on-premises deployments where data
must not leave the machine or network.

- **Why SrvDB:** never opens a network connection, stores everything in one
  local directory, logs no vector or metadata contents.
- **Start with:** whichever mode fits the data size.

## 5. Edge devices and small machines

Search on ARM boards, laptops and embedded Linux where memory and disk are
limited.

- **Why SrvDB:** prebuilt Linux aarch64 wheels; SQ8 stores vectors 4×
  smaller, PQ up to 32× smaller.
- **Start with:** `sq8`; measure recall against `flat` before trying `pq`.

## 6. Code search

Find functions by what they do, not their names, across a local codebase,
using a code embedding model.

- **Why SrvDB:** index a repository locally and keep it current with
  `upsert` and `delete` as files change.
- **Start with:** `flat` or `hnsw`.

## 7. Image, audio and multimodal similarity

Search photos by text ("sunset over water") or find similar images, with
CLIP-style embeddings; the same works for audio embeddings.

- **Why SrvDB:** any embedding with 128 to 4096 dimensions works, so the
  text and image vectors of a multimodal model share one database.
- **Start with:** `flat`; `hnsw` for large photo libraries.

## 8. Deduplication and near-duplicate detection

Find documents, support tickets or records that say the same thing in
different words.

```python
for doc_id, vector in items:
    hits = db.search(vector, k=2)          # the best hit is the item itself
    if len(hits) > 1 and hits[1][1] > 0.95:
        print(doc_id, "looks like", hits[1][0])
```

- **Why SrvDB:** scores are true cosine similarities, so a threshold means
  the same thing in every mode.
- **Start with:** `flat` for exact scores.

## 9. Recommendations ("more like this")

Recommend articles, products or songs similar to what a user just viewed,
from a local catalogue.

- **Why SrvDB:** a single `search` call on the item's own vector;
  `search_batch` for many items at once.
- **Start with:** `hnsw` for large catalogues.

## 10. Testing and prototyping RAG pipelines

Run retrieval in unit tests, notebooks and CI without starting a database
service.

- **Why SrvDB:** a database is just a temporary directory; tests stay fast
  and isolated.
- **Start with:** `flat`, so results are exact and repeatable.

## 11. One database per user or tenant

Desktop apps and services that keep each user's data apart.

- **Why SrvDB:** each database is a separate directory; deleting a user
  means deleting their directory.
- **Note:** each directory can be open in one place at a time, so keep one
  `Database` object per user in a single process.

## When to choose something else

SrvDB is deliberately small. Look elsewhere if you need:

- **A shared database server** used by many machines or services: SrvDB is
  a library, single-node, with no network access.
- **Several processes writing to the same database:** only one can open it
  at a time.
- **Filtering by metadata inside the search** (for example
  `where year > 2020`): planned for 0.4; today, filter the results after
  searching or keep one database per category.
- **Billions of vectors** on one machine: SrvDB has no published
  benchmarks at that scale yet.

## Live examples

Runnable, end-to-end examples and cheat sheets for these use cases are
[coming soon](/srvdb/examples/overview/).
