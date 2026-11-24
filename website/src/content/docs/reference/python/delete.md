---
title: srvdb.delete
description: Delete a database from disk.
---

Delete a database from disk: every SrvDB file, then the directory.

```python
srvdb.delete(path: str) -> None
```

## Parameters

- **path** : *str*
  Directory of a closed database.

## Returns

`None`.

## Raises

- **FileNotFoundError**
  `path` does not exist or is not a directory.

- **ValueError**
  `path` is a symbolic link (pass the real directory), or holds files SrvDB did not create (the message names one), a subdirectory, or no SrvDB database at all. Nothing is deleted.

- **BlockingIOError**
  The database is open, in this process or another. Close it first.

- **OSError**
  A file could not be removed.

## Notes

- Safety: only a directory that is provably a SrvDB database is removed. Every entry must be a file SrvDB writes, and at least one must start with a SrvDB header (magic number). A mistyped path that points at your home directory or a project folder is refused, and so is a symbolic link, even one that points at a database.
- With the database locked, the directory is first renamed to a hidden sibling `.<name>.srvdb-deleting-<pid>`, then emptied and removed, so no other process can open the database while it is being deleted. A crash part way can leave that hidden directory; remove it by hand. Deleted data cannot be recovered.
- To delete vectors rather than the whole database, use [`db.delete`](/srvdb/reference/python/db-delete/).

## Examples

```python
import os
import srvdb

with srvdb.create("./vectors", dimension=128) as db:
    db.add(["a"], [[1.0] + [0.0] * 127], ["{}"])
srvdb.delete("./vectors")
print(os.path.exists("./vectors"))
# False
```

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
try:
    srvdb.delete("./vectors")
except BlockingIOError:
    print("close it first")
# close it first
db.close()
srvdb.delete("./vectors")
```

## See also

- [`srvdb.create`](/srvdb/reference/python/create/): Create a new database with its index and options.
- [`db.close`](/srvdb/reference/python/db-close/): Release the database and its lock.
