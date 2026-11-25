---
title: db.memory_report
description: Return the estimated resident memory of the database, per part, and its memory budget.
---

Return the estimated resident memory of the database, per part, and its
memory budget.

```python
db.memory_report() -> dict
```

## Parameters

None.

## Returns

`dict` with these keys:

| Key | Meaning |
|-----|---------|
| `mode` | Index mode in use, such as `"flat"` or `"rabitq"` |
| `bits` | RaBitQ bits per dimension, or `None` |
| `dimension` | Vector dimension |
| `count` | Stored records, deleted ones included |
| `vector_bytes` | Vectors or codes, as searches read them |
| `index_bytes` | Graph, rotation, codebooks, quantizer or centroids |
| `links_bytes` | Links between records (approximate) |
| `keyword_bytes` | Keyword index, if loaded (approximate) |
| `total_bytes` | Sum of the four parts above |
| `budget_bytes` | `memory_mb` in bytes, or `None` without a budget |
| `limit_bytes` | 80 % of the budget, or `None` |
| `capacity` | Records the current mode holds within the limit, or `None` |
| `exact` | `True` when searches are exact (flat) |

## Raises

- `ValueError`: the database is closed.

## Notes

- These are estimates from the storage layouts, not readings from the
  operating system. See [Memory budget](/srvdb/concepts/memory-budget/).
- Thread safety: takes a shared lock, like [`db.info`](/srvdb/reference/python/db-info/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128, index="auto", memory_mb=50)
db.add(["a"], [[1.0] + [0.0] * 127], ["{}"])
report = db.memory_report()
print(report["mode"], report["vector_bytes"])
# flat 512
```

## See also

- [`srvdb.create`](/srvdb/reference/python/create/): the `memory_mb` option.
- [Fit a memory budget](/srvdb/guides/memory-budget/).
