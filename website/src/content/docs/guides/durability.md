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
  a mix. They carry a versioned header with a CRC32C checksum.
- Vector files are covered by CRC32C checksums, one per 1 MiB block, kept in
  a `<file>.crc` file next to each vector file. Every block is verified when
  the database opens, so damage that happened on disk (bit rot, a torn write,
  a bad copy) raises an error instead of returning wrong results. Copy the
  `.crc` files along with the rest of the directory when you back up.
  Design: [ADR 0007](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0007-checksums.md).

- The index mode and its parameters are stored in `index_settings.bin`
  (checksummed, replaced atomically). It is written when a database is
  created and rewritten by `persist()` when the mode or a parameter changed;
  a crash leaves the old or the new settings, never a mix, and never affects
  committed vectors. Opening with settings that conflict with the stored ones
  is refused with an error and changes nothing.
  Design: [ADR 0011](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0011-persisted-index-config.md).

Every change to the write path ships with a crash test that kills a process
mid-write, reopens, and checks the guarantee. Full design:
[ADR 0003](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0003-durability.md).

## Upgrading from 0.2

Opening a 0.2 database upgrades `metadata.db` to the current redb format,
after which 0.2 cannot open it. **Back up the directory first.**

## Upgrading from 0.3

Opening a 0.3 database adds checksums: each vector file gets a `.crc` file and
its header moves to format version 5, after which 0.3 cannot open it. Data
already on disk is checksummed as found; it is verified on every later open.
**Back up the directory first.**

A 0.3 database has no `index_settings.bin`. It opens as before (flat unless a
mode is passed); the settings are recorded the first time it is opened with an
explicit mode.

## When a checksum check fails

Opening a database verifies every vector file against its `.crc` file. If a
check fails, the open is refused (`RuntimeError` in Python,
`ChecksumMismatch` or `Corrupt` in Rust) and nothing is changed on disk. The
message names the file.

- **Damaged `hnsw.graph` or `ivf.index`**: delete the file. A damaged HNSW
  graph is already rebuilt automatically; IVF is retrained by the next
  `train_ivf()`.
- **Damaged vector file** (`vectors.bin`, `sq8_vectors.bin`,
  `quantized_vectors.bin`): the vectors themselves are damaged. Restore the
  whole database directory from a backup.
- **Missing or damaged `.crc` file** next to an intact vector file (for
  example, a backup that skipped `*.crc` files): restore the directory from a
  complete backup. A supported way to rebuild checksums is tracked in
  [issue #33](https://github.com/Srinivas26k/srvdb/issues/33).
