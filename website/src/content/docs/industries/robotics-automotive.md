---
title: Robotics & automotive
description: How SrvDB powers On-device Memory and other local AI tools in robotics & automotive. Use cases, architecture and code pattern.
sidebar:
  order: 12
---

Robots, vehicles and devices increasingly carry AI models that need memory, without a cloud round-trip.

:::note[Coming soon]
A complete, runnable **On-device Memory** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Cloud lookups add latency and fail without connectivity.
- Devices lose power unexpectedly.
- Storage and memory are limited.

## How SrvDB helps

- **In-process search:** no network hop.
- **Crash-safe:** after `persist()`, data survives power loss.
- **Compact storage** with SQ8 or PQ; ARM Linux wheels; usable from Rust.

## Use cases by level

### Level 1 · Find

*Semantic search over your own documents. One database, one `search` call.*

- Owner's-manual search in the vehicle
- Diagnostic trouble-code lookup
- Robot skill-library search
- Maintenance history search
- Part lookup
- Command-intent lookup

### Level 2 · Ask

*Grounded assistants: retrieve passages with SrvDB, answer with a local LLM, always cite sources.*

- In-car assistant over the owner's manual
- Technician assistant
- Robot operator Q&A
- Fleet operations Q&A
- Safety procedure Q&A
- Training assistant

### Level 3 · Match

*Similarity, deduplication and anomaly detection using cosine scores and thresholds.*

- Similar sensor episodes
- Telemetry-window anomalies
- Scene retrieval from dashcam embeddings
- Duplicate fleet bug reports
- Failure-signature matching
- Driving-behaviour patterns

### Level 4 · Act

*Agents with memory, several databases, multimodal embeddings and continuous `upsert` pipelines.*

- Robot episodic memory
- Place recognition with image embeddings
- Skill retrieval for task planning
- Fleet learning by sharing database snapshots
- Voice-assistant memory
- Multimodal maps

### Level 5 · Frontier

*Air-gapped, edge and research-grade systems. These usually need specialised embedding models and careful evaluation; some are research directions, not finished products.*

- On-device crash-safe memory through power loss
- Rust integration in robot software stacks
- Lifelong memory with periodic `compact()`
- Edge-case scenario mining for autonomous-driving tests
- Compressed indexes on embedded ARM
- Offline operation underground, underwater or in orbit

## Example: On-device Memory

An embedded memory component that stores what a robot or in-vehicle assistant has seen or been told, and recalls the most relevant items in the moment.

**How it works:**

1. Embed observations or utterances on the device.
2. Recall with `search` when context is needed.
3. `persist()` at safe points; after a power cut the memory reopens intact.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

memory = srvdb.Database("./device-memory", dimension=384, mode="sq8")

def recall(observation, k=3):
    return [json.loads(memory.get(i)) for i, _ in memory.search(embed(observation), k)]
```

**Recommended mode:** `sq8`; use the Rust API for firmware-adjacent services.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
