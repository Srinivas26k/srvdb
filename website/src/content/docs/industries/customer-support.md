---
title: Customer support
description: How SrvDB powers Helpdesk Answer Bot and other local AI tools in customer support. Use cases, architecture and code pattern.
sidebar:
  order: 7
---

Support teams answer the same questions in different words every day.

:::note[Coming soon]
A complete, runnable **Helpdesk Answer Bot** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Duplicate tickets waste agent time.
- Good past answers are buried in ticket history.
- Self-service bots give generic answers.

## How SrvDB helps

- **Suggested answers:** retrieve the closest solved tickets and knowledge-base articles.
- **Duplicate detection:** flag tickets that match an open one.
- **Runs inside your helpdesk app**, with no extra service to operate.

## Example: Helpdesk Answer Bot

An agent-side assistant that, for each new ticket, shows the three most similar solved tickets and the article that resolved them, and drafts a reply.

**How it works:**

1. Embed solved tickets and articles.
2. On each new ticket, search and show matches to the agent.
3. Upsert resolved tickets nightly and `persist()`.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

tickets = srvdb.Database("./tickets", dimension=384)

def suggested_answers(ticket_text, k=3):
    return [json.loads(tickets.get(i))["resolution"] for i, _ in tickets.search(embed(ticket_text), k)]
```

**Recommended mode:** `flat`; `hnsw` for very large ticket histories.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
