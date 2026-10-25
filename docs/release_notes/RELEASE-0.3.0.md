# SrvDB 0.3.0: Trustworthy storage

SrvDB 0.3 makes the database safe to rely on. Earlier versions could lose
or mix up data after a crash, some index modes did not do what their names
said, and every error looked the same. This release fixes that before any
new features are added.

**Before upgrading, back up your database directory.** Opening a 0.2
database upgrades `metadata.db`, after which 0.2 can no longer open it.
Read the [migration guide](../guides/migration.md) for every change that
needs action.

## Highlights

### Your data survives crashes

- `persist()` is a real commit point. Once it returns, every earlier add
  and delete survives a crash or power loss, and the database reopens in
  exactly that state. Later writes are discarded, never half-present.
- Fixed a crash bug where new vectors could be written after leftover
  bytes while ids were computed from the old count, so ids pointed at the
  wrong vectors.
- Index and quantizer files are replaced atomically and carry a versioned,
  checksummed header; every file is validated on open, and damaged files
  are reported instead of misread.

### Delete, upsert and compact

- `delete(ids)` removes vectors from every search mode; searches still
  return `k` results.
- `upsert(ids, ...)` inserts or replaces.
- `compact()` reclaims the disk space used by deleted vectors and is
  crash-safe.

### Search does what it says

- Scores are true cosine similarities in [-1, 1].
- `set_ef_search()` really changes HNSW search quality.
- `search_batch()` uses the active index instead of always scanning.
- A trained IVF index is reloaded on open instead of being lost.
- IVF trains with cosine (spherical k-means), matching how it searches.
- Auto mode builds an HNSW graph for large databases and never compresses
  vectors behind your back.
- HNSW over PQ storage (Rust) builds its graph; before, it returned
  nothing.

### Clear errors and safe concurrency

- Input is validated before anything is written: dimension, NaN or
  infinity, `k`, duplicate ids, JSON metadata.
- Python raises `ValueError`, `OSError`, `BlockingIOError` or
  `RuntimeError` according to the cause; Rust returns a typed `srvdb::Error`.
- A `Database` is safe to share between threads; every method releases the
  GIL and searches run in parallel. A second open of the same database is
  refused instead of risking corruption.

### Packaging

- The Python class is `srvdb.Database` (`SrvDBPython` still works with a
  deprecation warning until 0.4).
- Python 3.9-3.14. `numpy`, `psutil` and `scikit-learn` are no longer
  installed with SrvDB; the benchmark needs `pip install "srvdb[bench]"`.
- Releases include a software bill of materials (CycloneDX).

## Performance

This release publishes no performance numbers. Earlier tables measured
modes that did not work as labelled and have been removed. Future numbers
will come only from a committed, reproducible benchmark that lists the
hardware, OS, dataset, commit and command.

Full list of changes: [CHANGELOG.md](https://github.com/Srinivas26k/srvdb/blob/master/CHANGELOG.md).
