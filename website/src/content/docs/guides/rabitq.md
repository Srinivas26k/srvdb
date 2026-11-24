---
title: RaBitQ low-bit search
description: Training-free 1, 2 and 4-bit codes with exact rerank, and when to use them.
---

RaBitQ mode keeps every vector at full precision on disk and adds a small
code per vector: 1, 2 or 4 bits per dimension after a random rotation, plus
one correction factor. A search scans the codes to find `k × rerank`
candidates, then rescores those candidates exactly against the full
vectors. The method follows RaBitQ (Gao and Long, SIGMOD 2024) and its
multi-bit extension; see [ADR 0008](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0008-rabitq-quantization.md).

- **No training.** Unlike `sq8` and `pq`, nothing is learned from your data;
  the first vector can be added straight away.
- **Exact scores.** With rerank on (the default) the scores you get back
  are true cosine similarities.
- **Codes are derived data.** If the code files are missing, damaged or were
  made with other settings, SrvDB rebuilds them from the vectors on open.

## Using it

```python
import srvdb

db = srvdb.Database.new_rabitq("./db", dimension=384, bits=4)
db.add(["a", "b"], [[0.1] * 384, [0.2] * 384], ['{"t": 1}', '{"t": 2}'])
db.persist()
print(db.search([0.1] * 384, k=1))

db.set_rabitq_rerank(20)   # more candidates: higher recall, slower
db.set_rabitq_rerank(0)    # no rerank: estimated scores, fastest
```

The mode strings `"rabitq1"`, `"rabitq2"` and `"rabitq4"` work with
`srvdb.open(path, dimension, mode=...)` and `srvdb.Database(...)` too. Reopen a database with the
same bit width; opening it with another one rebuilds the codes.

From Rust:

```rust
use srvdb::{RabitqConfig, SrvDB};

let config = RabitqConfig::new(2)?.with_rerank(30);
let db = SrvDB::new_rabitq("./db", 384, config)?;
```

## Settings

| Setting | Values | Default |
|---------|--------|---------|
| `bits` | 1, 2 or 4 | 4 in Python |
| `rerank` | candidates per result; 0 turns rerank off | 40 (1 bit), 20 (2 bits), 10 (4 bits) |
| `seed` | rotation seed | fixed; changing it rebuilds the codes |

Fewer bits make the codes smaller and the scan faster but the estimates
rougher, so more candidates must be reranked for the same recall. Other bit
widths raise `ValueError` (Rust: `Error::InvalidInput`).

## When to use it

- **Flat search is too slow, and you want near-exact results** without
  building a graph. The scan reads 8 to 32 times fewer bytes than flat.
- **You cannot train a quantizer**: data arrives a few vectors at a time,
  or changes a lot.
- **Not to save disk space.** Full vectors are kept for reranking, so the
  directory is slightly larger than a flat one. Use `sq8` or `pq` when disk
  is the limit and approximate scores are acceptable.

Always measure recall against `flat` on a sample of your own data (see
[Choosing an index mode](/srvdb/guides/index-modes/)); tightly clustered
embeddings need a larger `rerank`. The benchmark harness runs the
`rabitq1`, `rabitq2` and `rabitq4` modes; see the
[benchmarks page](/srvdb/project/benchmarks/) for how to reproduce numbers.
