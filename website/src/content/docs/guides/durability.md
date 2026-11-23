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
  complete backup. If you have none and accept that the existing vectors can
  no longer be verified, you can make SrvDB rebuild the checksums from the
  data as found. Back up the directory, then mark the vector file as format
  version 4 (the 0.3 format, which has no `.crc` file). The next open then
  computes and saves new checksums:

  ```python
  from pathlib import Path

  path = Path("my_db/vectors.bin")  # or sq8_vectors.bin / quantized_vectors.bin
  with path.open("r+b") as f:
      f.seek(4)                      # header bytes 4-5: format version
      f.write((4).to_bytes(2, "little"))
  ```

  Any damage already in the vector file is then accepted as valid data, so
  use this only when restoring from a backup is not possible.
