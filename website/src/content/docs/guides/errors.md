---
title: Handling errors
description: Which exception SrvDB raises, and when.
---

| Exception | When |
|-----------|------|
| `ValueError` | Invalid input: wrong dimension, NaN or infinity, `k < 1`, empty or duplicate ids, metadata that is not a JSON object, unknown index or an option that does not apply to it, `create(exist_ok=True)` on a database with other settings, a call that does not fit the current mode, an add that would exceed the database's `memory_mb` budget even at 1 bit per dimension (nothing is written; see [Fit a memory budget](/srvdb/guides/memory-budget/)), use after `close()` |
| `FileNotFoundError` | `srvdb.open` or `srvdb.delete` found no database at the path |
| `FileExistsError` | `srvdb.create` found a database or other files at the path |
| `BlockingIOError` | The database is already open elsewhere |
| `OSError` | The operating system reported an I/O failure (disk full, permissions, ...) |
| `RuntimeError` | An internal fault, including a corrupt database file or data that fails its checksum (restore from a backup) |

Input is validated **before** anything is written, so a rejected call leaves
the database unchanged.

```python
try:
    db.add(ids, vectors, metas)
except ValueError as e:
    log.warning("rejected batch: %s", e)    # fix the input and retry
except OSError:
    raise                                    # disk problem: stop
```

Messages say what went wrong and how to fix it. They never contain your
vectors or metadata.

## Rust

The public API returns `srvdb::Result<T>` with a typed `srvdb::Error`:

```rust
match db.search(&query, 10) {
    Ok(hits) => { /* ... */ }
    Err(srvdb::Error::DimensionMismatch { expected, actual }) => { /* ... */ }
    Err(e) => return Err(e.into()),
}
```

Variants: `InvalidInput`, `DimensionMismatch`, `InvalidState`, `Locked`,
`Corrupt`, `ChecksumMismatch`, `SettingsMismatch`, `NotFound`,
`AlreadyExists`, `MemoryBudget`, `Io`, `Internal`. `ChecksumMismatch` means
a file was damaged after it was written; the message names the file and
byte range. `MemoryBudget` means an add to a database created with
`memory_mb` would not fit even at 1 bit per dimension; nothing was
written, and the message gives the bytes needed and the limit. Create the
database with a larger budget, `compact()` after deletes, or use a fixed
index mode.

If open fails only because a checksum file (`vectors.bin.crc` and the
like) is missing or damaged, for example after copying just `vectors.bin`,
close the database and call `SrvDB::rebuild_checksums(path)` (Python:
`srvdb.rebuild_checksums(path)`). It recomputes those files from the data
as found, so it cannot detect damage already in the data. If a checksum
file is intact but the data does not match it, the call refuses with
`ChecksumMismatch` and changes nothing: restore from a backup.
