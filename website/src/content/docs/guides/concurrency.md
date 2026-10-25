---
title: Threads and processes
description: Sharing a database between threads, and why a second open is refused.
---

## Threads

A `Database` object is safe to share between threads, with no lock of your
own:

- every method **releases the GIL**, so other Python threads keep running;
- **searches run in parallel**;
- writes (`add`, `upsert`, `delete`, `persist`, `compact`) wait for each
  other and for running searches instead of failing.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(lambda q: db.search(q, 10), queries))
```

For many queries at once, `search_batch(queries, k)` parallelises inside Rust.

## Processes

A database directory can be open in **one place at a time**. A second open,
from this process or another, raises `BlockingIOError`:

```python
a = srvdb.Database("./db", dimension=384)
b = srvdb.Database("./db", dimension=384)   # BlockingIOError: already open
```

Share one `Database` object instead of opening the same path twice. The lock is
released when the object is garbage-collected or the process exits, even after
a crash. For multiple processes, put one process in charge of the database and
send it requests.

## Rust

`SrvDB` is `Send + Sync`. Searches take `&self`; writes take `&mut self`.
Wrap it in `Arc<RwLock<SrvDB>>` to mix readers and a writer across threads.
Design: [ADR 0005](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0005-concurrency.md).
