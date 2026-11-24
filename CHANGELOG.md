# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LangChain and LlamaIndex vector-store adapters,
  `srvdb.integrations.langchain.SrvDBVectorStore` and
  `srvdb.integrations.llamaindex.SrvDBVectorStore`, over `srvdb.Database`:
  add texts or nodes with embeddings and metadata, similarity search with
  cosine scores, delete, and metadata filters (applied to search results).
  Install with `pip install "srvdb[langchain]"` or `"srvdb[llamaindex]"`;
  `import srvdb` never imports either framework, and the adapters make no
  network calls. See the "LangChain and LlamaIndex" guide (#26).
- `srvdb.open(path, dimension=1536, mode="flat")`, a shorter way to create
  or open a database. It returns a `srvdb.Database` and takes the same
  arguments as the `Database` constructor, which stays public and unchanged.
- CRC32C checksums over vector files (one per 1 MiB block, in a
  `<file>.crc` sidecar) and over index and quantizer files. Every vector file
  is verified when the database opens; damage (bit rot, a torn write, a bad
  copy) now fails with the new `Error::ChecksumMismatch` (Python
  `RuntimeError`) instead of returning wrong results. A damaged `hnsw.graph`
  is rebuilt from the verified vectors (ADR 0007).
- Benchmark harness (`cargo run --release -p srvdb --example bench`) that
  measures recall@10 against exact search, QPS, p50/p99 latency, build time
  and peak memory for the flat, SQ8, PQ, HNSW and IVF modes. It uses a seeded
  synthetic dataset by default (works offline) or user-supplied `.fvecs`
  files, and records CPU, OS, commit and command with every result. See
  `docs/benchmarks.md`.

### Changed
- On-disk format: vector file header version 5 and file envelope version 2.
  Databases from 0.3 open unchanged and are upgraded on open; after that,
  0.3 can no longer open them. Back up `*.crc` files with the rest of the
  database directory. Opening reads every vector file once, so open time
  grows with data size.
- SQ8 search is about 2x faster: each code is scored in one pass without
  allocating a decoded vector, and only the top k results are selected instead
  of sorting every score. Scores agree with the previous implementation to
  within floating-point rounding; the on-disk format is unchanged.
- SQ8 flat search now scans codes in parallel, as flat f32 search does, and
  scores each code with a wider, precomputed decode step. On 5000 vectors it
  went from about a quarter of flat search throughput to faster than flat
  (see `docs/benchmarks.md` to reproduce). Scores agree with the previous
  implementation to within 1e-5; the on-disk format is unchanged (#29).

### Removed
- The deprecated `srvdb.SrvDBPython` alias (deprecated in 0.3). Migrate by
  replacing `srvdb.SrvDBPython(...)` with `srvdb.Database(...)`; arguments are
  unchanged (#26).

### Fixed
- HNSW recall on clustered data (#28). The graph kept only the M nearest
  neighbours per node, which dropped every link between clusters, so a search
  could only reach the entry point's cluster (recall@10 about 0.70 on 5000
  clustered vectors). Neighbour selection and pruning now use the paper's
  diversity heuristic, and recall@10 is 1.0 on the same benchmark. Search
  visits more of the graph, so HNSW queries and builds are slower than the
  broken version. The `hnsw.graph` format is unchanged. A graph saved by an
  older version still loads but keeps its poor links: delete `hnsw.graph`
  to rebuild it from the stored vectors.

## [0.3.0] - 2026-10-25

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
- New documentation website at https://srinivas26k.github.io/srvdb/
  (Astro Starlight), with guides, a local RAG walkthrough and API
  reference; it replaces the MkDocs pages.
- SrvDB has a logo: the cubist eagle (`assets/brand/`).
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
