---
title: Handling errors
description: Which exception SrvDB raises, and when.
---

| Exception | When |
|-----------|------|
| `ValueError` | Invalid input: wrong dimension, NaN or infinity, `k < 1`, empty or duplicate ids, metadata that is not a JSON object, unknown index or an option that does not apply to it, `create(exist_ok=True)` on a database with other settings, a call that does not fit the current mode, use after `close()` |
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
`Corrupt`, `ChecksumMismatch`, `Io`, `Internal`. `ChecksumMismatch` means a
file was damaged after it was written; the message names the file and byte
range.
