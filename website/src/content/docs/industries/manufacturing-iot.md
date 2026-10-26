---
title: Manufacturing & IoT
description: How SrvDB powers Factory-floor Maintenance Assistant and other local AI tools in manufacturing & iot. Use cases, architecture and code pattern.
sidebar:
  order: 6
---

Factories and field sites need answers at the machine, often on edge hardware with poor connectivity.

:::note[Coming soon]
A complete, runnable **Factory-floor Maintenance Assistant** example is in progress. The pattern below already works with SrvDB 0.3.
:::

## The problem

- Technicians search thick manuals for the right procedure.
- Past fault reports that match today's symptoms are hard to find.
- Edge devices have little memory and unreliable networks.

## How SrvDB helps

- **Manual and fault-report search** on the device itself.
- **Small storage:** SQ8 stores vectors 4× smaller, PQ up to 32× smaller.
- **Linux aarch64 wheels** for ARM edge hardware; works with no connectivity.

## Example: Factory-floor Maintenance Assistant

A tablet or edge-box assistant where a technician describes a symptom and gets the closest past fault reports and the relevant manual pages.

**How it works:**

1. Embed manuals and fault reports on a workstation, then copy the database folder to the device.
2. On the device, search by symptom description.
3. Add new fault reports locally and `persist()`.

## Code pattern

`embed()` is your embedding model (for example sentence-transformers).

```python
import json
import srvdb

faults = srvdb.Database("./faults", dimension=384, mode="sq8")

def similar_faults(symptom, k=5):
    return [json.loads(faults.get(i)) for i, _ in faults.search(embed(symptom), k)]
```

**Recommended mode:** `sq8` on small devices; check recall against `flat` first.

## Learn more

- [Use cases](/srvdb/start/use-cases/)
- [Build a local RAG app](/srvdb/guides/local-rag/)
- [Durability and persist()](/srvdb/guides/durability/)
- [Live examples & cheat sheets](/srvdb/examples/overview/) (coming soon)
