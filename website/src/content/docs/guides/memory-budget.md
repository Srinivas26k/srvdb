---
title: Fit a memory budget
description: Create a SrvDB database that stays within a RAM budget, watch it move from flat to RaBitQ as it grows, and handle the error when the budget is full.
---

Use a memory budget when the database must share a small machine, such as
a laptop app or an edge device, and you would rather trade a little
search precision than run out of RAM. How the estimate works is described
in [Memory budget](/srvdb/concepts/memory-budget/).

## Create a budgeted database

```python
import srvdb

db = srvdb.create("./notes", dimension=64, index="auto", memory_mb=1)
report = db.memory_report()
print(report["mode"])
# flat
print(report["budget_bytes"])
# 1048576
```

An empty database always starts flat, which is exact. `capacity` tells
you how many records the current mode can hold before the next step.

## Watch it step down

```python
import random

import srvdb

db = srvdb.create("./notes", dimension=64, index="auto", memory_mb=1)
capacity = db.memory_report()["capacity"]

rng = random.Random(0)
vectors = [[rng.uniform(-1, 1) for _ in range(64)] for _ in range(capacity + 1)]
ids = [f"doc{i}" for i in range(len(vectors))]
db.add(ids, vectors, ["{}"] * len(vectors))

report = db.memory_report()
print(report["mode"], report["bits"])
# rabitq 4
print(report["total_bytes"] <= report["limit_bytes"])
# True
db.save()
```

The add that crossed the flat capacity re-encoded every stored vector to
4-bit RaBitQ first. Searches still rerank the best candidates against the
full vectors. The step happens at most three times (4, 2, then 1 bit) and
is recorded, so `srvdb.open(path)` reopens in the same mode with the same
budget.

## When the budget is full

Past 1-bit capacity, `add` raises `ValueError` and writes nothing:

```python
import random

import srvdb

db = srvdb.create("./tiny", dimension=64, index="auto", memory_mb=1)
rng = random.Random(1)
too_many = [[rng.uniform(-1, 1) for _ in range(64)] for _ in range(60_000)]
message = ""
try:
    db.add([f"d{i}" for i in range(60_000)], too_many, ["{}"] * 60_000)
except ValueError as e:
    message = str(e)
print("memory_mb" in message)
# True
print(db.count())
# 0
```

To go on, create a new database with a larger `memory_mb`, call
`compact()` if many records are deleted (deleted records use memory
until then), or use a fixed index mode.

## Check any database

`db.memory_report()` also works without a budget: it estimates the
footprint of any mode, which helps when you choose one.
