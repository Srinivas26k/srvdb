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

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Knowledge-base search
- Ticket history search
- Macro search
- Product documentation search
- Release-note search
- Community forum search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Answer bot grounded in the knowledge base
- Agent-assist reply drafting
- New-agent onboarding tutor
- Multilingual answers
- Escalation-guide Q&A
- Internal IT helpdesk bot

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Duplicate tickets
- Topic clustering
- Unusual-issue detection by distance
- Knowledge-base gaps: tickets with no close article
- Macro recommendation
- Churn-risk conversations similar to past churned customers

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Agent with per-customer history memory
- One database per product
- Screenshot embeddings to match UI bugs
- Continuous knowledge-base updates
- Voice-call transcript search
- Per-tenant databases for SaaS products

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- Fully offline support kiosks
- Early incident detection from bursts of tickets near the same vector
- Self-improving knowledge-base loop
- On-device assistants inside appliances
- Cross-language issue linking
- Crash-safe conversation memory

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
