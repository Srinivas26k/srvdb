---
title: Architecture
description: How SrvDB is put together.
---

```
 Python app ──► srvdb (Python package)
                  │  srvdb._low_level  (PyO3 bindings: argument conversion,
                  │                     error mapping, GIL release)
                  ▼
               srvdb (Rust core crate)
                  ├─ engine/    open, write, read, delete, commit, compact, modes
                  ├─ storage/   vector files, header, metadata (redb), index files, lock
                  ├─ index/     HNSW graph, IVF partitions, hybrid search
                  ├─ quant/     SQ8 and PQ quantization
                  ├─ distance/  SIMD cosine kernels and scans
                  └─ tuning/    auto mode
```

Principles:

- **The core knows nothing about Python.** Bindings stay thin; validation
  lives in the core so Rust users get it too.
- **Committed data first.** `redb` holds the commit record; vector files are
  append-only and truncated to the commit on open.
- **Indexes are derived data**, always rebuildable from the vectors.
- **No network, no stdout.** Logging goes through the `log` crate.

Engineering rules: [AGENTS.md](https://github.com/Srinivas26k/srvdb/blob/master/AGENTS.md).
