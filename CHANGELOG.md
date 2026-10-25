# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Licence
- SrvDB is now licensed under the Apache License 2.0 (was AGPL-3.0), so it
  can be used in open-source and commercial products alike (ADR 0006).
  Releases up to 0.2.0 remain under AGPL-3.0.

### Changed
- The Python class is now `srvdb.Database` (was `srvdb.SrvDBPython`). The
  old name still works but emits a `DeprecationWarning` and will be removed
  in 0.4.
- The compiled extension moved to the private module `srvdb._low_level`;
  import from `srvdb` as before.
- Rust: `srvdb::core::*` paths were removed; use `srvdb::types::*`.
- Rust: the public API returns `srvdb::Result` with a typed `srvdb::Error`
  instead of `anyhow::Error` (see ADR 0002).
- Inputs are validated before anything is written: vectors must match the
  database dimension and contain no NaN or infinity, `k` must be at least 1,
  vector and metadata counts must match, and metadata must be valid JSON
  (Rust callers that stored plain strings must now wrap them, for example
  `{"id": "doc1"}`).
- Python raises `ValueError` for invalid input (wrong dimension, NaN or
  infinity, `k < 1`, empty or duplicate ids, metadata that is not a JSON
  object), `OSError` for I/O failures and `RuntimeError` otherwise.
  Previously every failure was `RuntimeError`, and invalid metadata was
  silently stored as `{}`.
- A Rust panic inside SrvDB now raises a Python `PanicException` instead of
  terminating the interpreter.
- Library logging goes through the `log` crate; nothing is printed to
  stdout or stderr.

### Added
- Delete that really deletes: `SrvDB::delete(id)` (Rust) and
  `Database.delete(ids)` (Python) tombstone vectors so no search returns
  them, and searches still return `k` results. Previously Python's delete
  only hid the id while the vector kept occupying top-k slots.
- `SrvDB::replace(id, vector, metadata)` and `Database.upsert(ids, ...)`:
  existing ids point at the new vector, new ids are added.
- `SrvDB::live_count()`; Python `count()` now counts searchable vectors.
- `compact()` (Rust and Python) removes deleted vectors from disk and
  renumbers internal ids; Python string ids and search results are
  unchanged. Crash-safe: an interrupted compaction is finished or
  discarded on the next open.
- Deletes follow the same durability rules as adds: made durable by
  `persist()`, undone on reopen otherwise (ADR 0004).

### Concurrency
- A database can be open in only one place at a time: a second open (same
  or another process) fails with `Error::Locked` / Python
  `BlockingIOError` instead of risking corruption. The lock is released
  automatically when a process exits, even on a crash (ADR 0005).
- `SrvDB` is `Send + Sync`: searches run in parallel through `&SrvDB`.
- A Python `Database` is safe to share between threads without your own
  lock: every method releases the GIL, searches run in parallel, and
  writes wait for each other instead of failing.
- Building from source requires Rust 1.99 or newer (wheels from PyPI need no
  Rust toolchain).

### Durability
- `persist()` is now a real commit point (ADR 0003): everything added
  before it returns survives a crash or power loss, and a database
  reopens exactly as of the last successful `persist()`. Writes made after
  it are discarded on reopen instead of being half-present.
- Searches see added vectors immediately, without `persist()`.
- Fixed: after a crash, new vectors could be written behind leftover bytes
  while ids were computed from the old count, so ids pointed at the wrong
  vectors.
- Index and quantizer files are replaced atomically (temp file, fsync,
  rename).
- Index and quantizer files carry a versioned, checksummed header; damaged
  or foreign files are reported as corrupt instead of being misread. The
  unmaintained `bincode` dependency is gone. v0.2 quantizer files still
  load, and v0.2 index files are rebuilt automatically.
- `metadata.db` now uses the redb 4 file format. A database created by
  SrvDB 0.2 is upgraded automatically on first open (after which 0.2 can
  no longer open it).

### Fixed
- HNSW with PQ storage (`new_with_hnsw_quantized`) now builds its graph;
  it indexed nothing before, so every search returned no results. The
  graph is saved by `persist()`, and calling the constructor on an
  existing database reopens it with its codebooks instead of failing.
- Auto mode now only chooses an index: above the size threshold it builds
  an HNSW graph (saved by `persist()` and reloaded on open). It no longer
  converts storage to SQ8 behind the caller's back, and the mode stays
  `Auto` after the decision.
- A trained IVF index is reloaded when the database is opened; it used to
  be saved by `persist()` but never read back, so it had to be retrained
  after every restart.
- `search_batch()` uses the active index (HNSW, IVF, SQ8, PQ) like
  `search()`; it used to scan every vector exactly, ignoring the index.
- `set_ef_search()` now changes HNSW search quality (it edited a copy that
  search never read); it applies to IVF partitions too, is saved with the
  index, rejects 0 and is available from Python.
- Search scores are true cosine similarities in [-1, 1]. Flat (and HNSW
  and IVF) search used to report (1 + cos) / 2, so an orthogonal vector
  scored 0.5 and scores differed in scale from SQ8 search. Result order is
  unchanged; code with score thresholds should be checked.
- HNSW inserts from multiple threads (used to fill IVF partitions) could
  leave vectors unfindable and could panic.
- IVF searches no longer hide errors from individual partitions.
- Training on vectors containing NaN no longer panics.

### Packaging
- Requires Python 3.9 or newer (3.8 is end-of-life); wheels for 3.9-3.14.
- `numpy`, `psutil` and `scikit-learn` are no longer installed with
  `srvdb`; the library never used them. The benchmark script needs them:
  `pip install "srvdb[bench]"`.
- The benchmark script moved to `benchmarks/universal_benchmark.py`;
  one-off debug scripts were removed from the repository root.

### Documentation
- Performance tables from earlier versions were removed: they measured
  modes that did not work as labelled and cannot be reproduced. Numbers
  will be published only from a committed, reproducible benchmark.
- README and the modes guide describe 0.3 as shipped; the migration guide
  lists every change that needs action when upgrading.

## [0.2.0] - 2025-12-30

### 🚀 Major Features
- **Dynamic Dimensions Support**: SrvDB now supports vector dimensions from 128 to 4096 (determined at runtime) instead of compile-time fixed dimensions.
    - Updated `Vector`, `VectorStorage`, and `QuantizedVectorStorage` to use `Vec<f32>`/`&[f32]`.
    - Removed `const N: usize` generic parameters throughout the codebase.
- **True SQ8 Disk Compression**:
    - Refactored `ScalarQuantizedStorage` to store vectors as `u8` bytes on disk.
    - Achieves **4x raw compression** (32-bit float → 8-bit integer) compared to uncompressed storage.
    - Observed ~2.6x to ~4x effective reduction in total index size depending on metadata overhead.
    - Maintains **100% recall** on tested datasets.
- **HNSW Indexing Integration**:
    - Added initial support for HNSW graph-based indexing for faster search (O(log n)).
- **IVF-HNSW Hybrid Indexing**:
    - Implemented **Inverted File System (IVF)** with HNSW refinement (`IVFIndex`).
    - Uses **K-Means Clustering** to partition vector space into Voronoi cells.
    - Two-stage search: **Coarse Search** (Centroids) + **Fine Search** (HNSW within partitions).
    - Scalable to **100k+ vectors** with sub-20ms latency and minimal memory overhead.
    - Python API: `db.set_mode("ivf")`, `db.configure_ivf()`, `db.train_ivf()`.

### 🐛 Bug Fixes
- Fixed critical bug where SQ8 storage was wrapping explicit `f32` floats, negating compression benefits.
- Fixed `panic` in `select_top_k` when candidate set was empty.
- Fixed routing logic in `SrvDB::add`, `SrvDB::search`, and `SrvDB::persist` to correctly handle `ScalarQuantizedStorage`.
- Fixed multiple compilation errors related to type mismatches and missing imports during the v0.2.0 upgrade.
- Fixed Python binding initialization for SQ8 mode (`new_scalar_quantized`).
- Fixed **HNSW Panic** where inserting a node into a new top layer attempted to link to non-existent nodes.
- Fixed persistence race condition where training data was not visible due to missing flush.

### ⚡ Performance
- **Ingestion**: Optimized `add_batch` implementation for SQ8, achieving ~25k vectors/sec on standard hardware.
- **Zero-Copy**: Leveraged memory mapping (`mmap`) for instant startup and low memory footprint in SQ8 mode.

### 🛠 breaking Changes
- `SrvDB::new` and related constructors now require a `dimension` argument (or infer it from config).
- Python API updated to accept `dimension` in constructor.
- Data format change: v0.2.0 storage format is not backward compatible with v0.1.x due to dynamic dimension header changes.

## [0.1.9] - Previous Versions
- Initial release with Flat Indexing.
- Basic Python bindings.
- Fixed-dimension support.
