---
title: Upgrading
description: Every change in 0.3 and 0.4 that may need action.
---

## 0.4: simple verbs

The Python API is a few verbs ([ADR 0013](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0013-simple-python-verbs.md)).
Code written for 0.3 keeps working unchanged; new code uses:

| Task | 0.4 |
|------|-----|
| Create a database (any index, with options) | `srvdb.create(path, dimension, index="hnsw", **options)` |
| Reopen it | `srvdb.open(path)`: dimension, index and options are read from disk |
| Commit writes | `db.save()` (`db.persist()` is an alias) |
| Release the lock | `db.close()`, or a `with` block, which also saves |
| Remove it from disk | `srvdb.delete(path)` |

The 0.3 class constructor and its `new_*` constructors map to
`srvdb.create(path, dimension, index=..., **options)`; see
[`srvdb.create`](/srvdb/reference/python/create/) for the options of each index.

## 0.3

:::danger[Back up first]
Opening a 0.2 database upgrades `metadata.db` to a new file format, after
which 0.2 can no longer open it. Copy the database directory before upgrading.
:::

| Change | What to do |
|--------|------------|
| Python 3.9+ required | Upgrade Python 3.8 environments. |
| numpy, psutil, scikit-learn no longer installed | Add them to your own dependencies if you import them. |
| `srvdb.SrvDBPython` renamed (old name removed in 0.4) | Use `srvdb.create` and `srvdb.open`. |
| Scores are true cosine in [-1, 1] (were `(1 + cos) / 2` for flat, HNSW, IVF) | Convert thresholds: new = 2 × old − 1. Order is unchanged. |
| Metadata must be a JSON object | Wrap plain strings: `{"text": "..."}`. |
| Specific exceptions | Catch `ValueError` / `OSError` / `BlockingIOError` instead of only `RuntimeError`. |
| One open instance per database | Share one database object between threads. |
| `save()` is the commit point | Later writes are discarded after a crash; save at checkpoints. |
| `count()` excludes deleted vectors | — |
| Auto mode never compresses | Create with `index="sq8"` if you relied on it. |
| Licence is Apache-2.0 (was AGPL-3.0) | Nothing; you have more freedom. |

Full list: [CHANGELOG](https://github.com/Srinivas26k/srvdb/blob/master/CHANGELOG.md).
