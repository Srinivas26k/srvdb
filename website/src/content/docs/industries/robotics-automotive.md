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
