---
title: Healthcare
description: How SrvDB powers Clinical Notes Assistant and other local AI tools in healthcare. Use cases, architecture and code pattern.
sidebar:
  order: 2
---

Clinical text is among the most sensitive data there is, and hospitals often cannot send it to external AI services.

:::note[Coming soon]
A complete, runnable **Clinical Notes Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Clinicians spend time searching notes, guidelines and discharge summaries.
- Similar past cases are hard to find by keyword alone.
- Patient data must stay inside the hospital's own systems.

## How SrvDB helps

- **Guideline and protocol RAG:** answer questions from internal guidelines with a local LLM.
- **Similar-case search:** find de-identified past cases that resemble a current one.
- **On-premises only:** data stays in a local directory on hospital hardware.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Clinical protocol search by question
- Local formulary and drug-information search
- De-identified discharge summary search
- Terminology aid: closest code descriptions for free-text findings (clinician confirms)
- Laboratory SOP search
- Handover note search

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- Protocol Q&A with citations to the exact passage
- Patient-education assistant limited to approved leaflets
- Discharge summary drafting from retrieved templates (clinician edits)
- Clinical trial protocol Q&A for coordinators
- Clinical coding suggestions from descriptions (coder verifies)
- Research assistant over the hospital's own publications

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Similar-case retrieval over de-identified cases
- Duplicate patient-record candidates for data-quality teams
- Adverse-event report clustering
- Unusual-presentation flag: triage notes far from typical ones
- Trial pre-screening shortlist: patient summaries vs eligibility criteria (staff decide)
- Research image similarity on de-identified imaging sets

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Ward assistant with per-clinician session memory
- Combined lookup across guideline, formulary and local-policy databases
- Radiology research tool joining image and report embeddings
- Guideline updates via `upsert`, retention via `delete` and `compact()`
- Offline assistant on hospital tablets
- Pharmacovigilance pipeline matching new reports to known signals

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- On-premises hospital AI stack with zero data egress
- Genomics and protein embedding similarity for research (domain models)
- Offline assistants for ambulances and rural clinics
- Per-site databases: each hospital keeps its own data; only models or aggregates are shared
- Retrospective research on ICU waveform embedding similarity
- Crash-safe audit log of AI suggestions shown to clinicians

## Example: Clinical Notes Assistant

A ward-side assistant that answers questions from the hospital's own protocols and formularies, citing the passages it used, running entirely on hospital hardware.

**How it works:**

1. Embed protocols, formularies and (de-identified) case summaries.
2. For each question, retrieve the closest passages and show them with their source.
3. A local LLM drafts an answer from those passages only; the clinician decides.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

protocols = srvdb.Database("./protocols", dimension=384)

def sources_for(question, k=4):
    return [json.loads(protocols.get(i)) for i, _ in protocols.search(embed(question), k)]
```

**Recommended mode:** `flat`.

:::caution
SrvDB keeps data local, but it is not a certified medical device or a compliance product. Regulatory compliance (HIPAA, GDPR and similar) remains the deployer's responsibility.
:::

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
