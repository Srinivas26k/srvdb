---
title: srvdb.rebuild_checksums
description: Recompute missing or damaged checksum files of a closed database.
---

Recompute missing or damaged checksum files (`*.crc`) of a closed database, so it opens again.

```python
srvdb.rebuild_checksums(path: str) -> list[str]
```

## Parameters

- **path** : *str*
  Directory of a closed database.

## Returns

The names of the checksum files that were rewritten, such as `["vectors.bin.crc"]`. Empty if every checksum file was intact.

## Raises

- **FileNotFoundError**
  `path` is not a directory.

- **BlockingIOError**
  The database is open, in this process or another. Close it first.

- **OSError**
  A file could not be read or written.

- **RuntimeError**
  A checksum file is intact but the data does not match it, or the data is truncated: the data is damaged. Nothing is rewritten; restore from a backup.

## Notes

- It never runs on its own. Call it when `srvdb.open` fails because a `.crc` file is missing or damaged, for example after copying only `vectors.bin`.
- A checksum file that is missing, fails its own check, or describes another file is recomputed from the data as found. A rebuilt checksum cannot detect damage that was already in the data. If the data itself may be damaged, restore a backup instead.
- Each checksum file is replaced atomically (temporary file, fsync, rename). After a crash, run the call again. Vectors, metadata and indexes are not changed.

## Examples

```python
import os
import srvdb

with srvdb.create("./vectors", dimension=128, index="flat") as db:
    db.add(["a"], [[1.0] + [0.0] * 127], ["{}"])
os.remove("./vectors/vectors.bin.crc")
print(srvdb.rebuild_checksums("./vectors"))
# ['vectors.bin.crc']
with srvdb.open("./vectors") as db:
    print(db.count())
# 1
```

## See also

- [`srvdb.open`](/srvdb/reference/python/open/): Open an existing database.
- [Handling errors](/srvdb/guides/errors/)
