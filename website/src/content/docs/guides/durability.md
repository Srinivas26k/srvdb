---
title: Durability and persist()
description: Exactly what survives a crash, and how to use persist().
---

## The guarantee

> When `persist()` returns successfully, every earlier `add`, `upsert` and
> `delete` survives a process crash or power loss. After a crash the database
> reopens in exactly the state of the last successful `persist()`: later
> writes are absent, never partially present, and nothing committed is
> corrupted.

Writes after the last `persist()` may be lost. Searches see writes as soon as
the call that made them returns, before `persist()`.

## Using it

```python
for batch in batches:
    db.add(batch.ids, batch.vectors, batch.metas)
    db.persist()          # checkpoint: this batch is now safe
```

`persist()` flushes and fsyncs to disk, so it costs real time. Call it once
per batch or at natural checkpoints, not after every vector.

## How it works

- Vectors are appended to a file; metadata goes to an embedded
  [redb](https://github.com/cberner/redb) database.
- `persist()` fsyncs the vector file, then records the committed count and
  pending deletes in **one durable redb transaction**. That transaction is the
  commit point.
- On open, anything beyond the committed count is truncated, so the vector
  file and metadata always agree.
- Index and quantizer files are written to a temporary file, fsynced and
  renamed over the old one, so a reader sees the old or the new version, never
  a mix. They carry a versioned, checksummed header.

Every change to the write path ships with a crash test that kills a process
mid-write, reopens, and checks the guarantee. Full design:
[ADR 0003](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0003-durability.md).

## Upgrading from 0.2

Opening a 0.2 database upgrades `metadata.db` to the current redb format,
after which 0.2 cannot open it. **Back up the directory first.**
