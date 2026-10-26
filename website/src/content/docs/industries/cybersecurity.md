---
title: Cybersecurity
description: How SrvDB powers SOC Copilot and other local AI tools in cybersecurity. Use cases, architecture and code pattern.
sidebar:
  order: 1
---

Security teams drown in alerts, logs and threat reports, and much of that data is too sensitive to send to a cloud AI service.

:::note[Coming soon]
A complete, runnable **SOC Copilot** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Analysts re-investigate incidents that look like ones already solved.
- Phishing campaigns reuse wording with small changes that defeat exact-match rules.
- Threat intelligence lives in hundreds of reports nobody can search by meaning.
- Many SOCs run on isolated networks where cloud AI is not allowed.

## How SrvDB helps

- **Similar-incident search:** embed alert descriptions and investigation notes; a new alert instantly shows the closest past cases and how they were resolved.
- **Phishing similarity:** embed known phishing emails; a new message that scores above a cosine threshold against them is flagged, even if reworded.
- **Threat-intel RAG:** ask questions over reports and advisories with a local LLM.
- **Air-gapped by design:** SrvDB never opens a network connection and never logs vector or metadata contents.

## Example: SOC Copilot

An analyst assistant that runs on the SOC's own machines. When an alert fires it retrieves the most similar past incidents, the playbook steps used, and relevant threat-intel passages, then drafts a triage summary with a local LLM. Nothing leaves the network.

**How it works:**

1. Ingest closed incidents, playbooks and threat reports; store the text and ticket id as metadata.
2. On each new alert, embed it and `search` for the 5 closest incidents and passages.
3. Give them to a local LLM to draft triage notes; the analyst confirms.
4. When a case closes, `upsert` it so the next search learns from it; `persist()` after each batch.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

incidents = srvdb.Database("./soc-incidents", dimension=384)

def similar_incidents(alert_text, k=5):
    hits = incidents.search(embed(alert_text), k)
    return [(json.loads(incidents.get(i))["summary"], round(s, 2)) for i, s in hits]

def close_incident(ticket_id, summary):
    incidents.upsert([ticket_id], [embed(summary)], [json.dumps({"summary": summary})])
    incidents.persist()
```

**Recommended mode:** `flat` for exact, explainable matches; `hnsw` once the archive is large.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
