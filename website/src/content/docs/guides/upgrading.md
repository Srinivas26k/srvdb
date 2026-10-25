---
title: Upgrading to 0.3
description: Every change in 0.3 that may need action.
---

:::danger[Back up first]
Opening a 0.2 database upgrades `metadata.db` to a new file format, after
which 0.2 can no longer open it. Copy the database directory before upgrading.
:::

| Change | What to do |
|--------|------------|
| Python 3.9+ required | Upgrade Python 3.8 environments. |
| numpy, psutil, scikit-learn no longer installed | Add them to your own dependencies if you import them. |
| Class renamed to `srvdb.Database` | `srvdb.SrvDBPython` still works with a warning until 0.4. |
| Scores are true cosine in [-1, 1] (were `(1 + cos) / 2` for flat, HNSW, IVF) | Convert thresholds: new = 2 × old − 1. Order is unchanged. |
| Metadata must be a JSON object | Wrap plain strings: `{"text": "..."}`. |
| Specific exceptions | Catch `ValueError` / `OSError` / `BlockingIOError` instead of only `RuntimeError`. |
| One open instance per database | Share one `Database` between threads. |
| `persist()` is the commit point | Later writes are discarded after a crash; persist at checkpoints. |
| `count()` excludes deleted vectors | — |
| Auto mode never compresses | Open with `mode="sq8"` if you relied on it. |
| Licence is Apache-2.0 (was AGPL-3.0) | Nothing; you have more freedom. |

Full list: [CHANGELOG](https://github.com/Srinivas26k/srvdb/blob/master/CHANGELOG.md).
