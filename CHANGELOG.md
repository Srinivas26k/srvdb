# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Theorem-proving premise-retrieval example (#44) in
  `examples/theorem_proving/`. It ranks a bundled sample of 207 Lean
  mathlib lemma statements (Apache-2.0, attributed in `NOTICE`) for a given
  proof state, using vector, keyword (BM25) and hybrid search, `uses`
  dependency links with `expand=`, and a `where=` filter. It prints recall@k
  on 45 labelled proof steps together with the hardware and versions, and
  runs offline with TF-IDF + TruncatedSVD embeddings. The README shows how
  to swap in a local embedding model and how to evaluate on LeanDojo
  Benchmark 4 after downloading it yourself (`leandojo_to_tsv.py`).
  Example dependencies are in `examples/theorem_proving/requirements.txt`,
  not in the package. New guide: "Premise retrieval for theorem proving".
- Typed, directed links between records and k-hop search expansion (#80,
  ADR 0016). Rust: `SrvDB::link`, `unlink`, `neighbors`, `link_count`,
  `move_links` and `search_expand` with `Expansion` (1 to 3 hops, at most
  1,000 linked records) returning `ExpandedResult` with a hop distance.
  Python: `db.link(a, "cites", b)`, `db.unlink`, `db.neighbors(id,
  type=None)` and `db.search(q, k, expand=("cites", 2))`, which returns
  `(id, score, hop)` tuples. Links are stored in `metadata.db` and
  committed in the same transaction as records by `persist()`/`save()`;
  deleting a record removes its links, replace/upsert keep them, and
  compaction renumbers them. Databases without links open unchanged.
  Limits: 1,024 outgoing and 100,000 incoming links per record, 1,000,000
  links and 1,024 link types per database. If SrvDB 0.3 changes records
  after links were saved, opening refuses with `Corrupt` (`RuntimeError`)
  instead of using possibly wrong links; `SrvDB::drop_links(path)` /
  `srvdb.drop_links(path)` removes them.
- Certified and deadline search for RaBitQ (#81, ADR 0017).
  `db.search(q, k, certify=True)` gives every scanned vector an error
  interval from the RaBitQ bound plus the u8 table error. It certifies
  results whose interval clears the k-th lower bound and rescores only
  the uncertain candidates on the full vectors, at most
  `max(k * rerank, 1000)` of them. `confidence` (0.99) is a nominal,
  approximate bound on a probability over a hypothetical random rotation
  seed: it is not a per-query guarantee, not a long-run rate for the
  database's fixed seed, and not joint across queries. It is `None`
  when the rerank cap or the deadline cut the rescoring short. On
  clustered data almost nothing is certified, and 1-bit search can
  rescore far more candidates than a normal search. `deadline_ms=` stops the code scan between
  blocks, and the rescoring between batches, once the time is up and reports `completeness`. Both return a
  list-like `SearchOutcome`, and both raise `ValueError` outside the
  RaBitQ modes. Rust: `SrvDB::search_rabitq_with` with `RabitqSearch` (incl. `with_max_rerank`),
  `SearchOutcome` and `SearchStats`. RaBitQ memory grows by 4 bytes per
  vector (derived error widths; the file format is unchanged).
- Keyword and hybrid search (#79, ADR 0015). `SrvDB::enable_keyword_index(field)`
  indexes one string field of the metadata JSON (Unicode-aware tokenizer, no
  model or network); `SrvDB::keyword_search(text, k, filter)` ranks records by
  BM25 and `SrvDB::search_hybrid(query, text, k, filter)` fuses vector and
  keyword rankings with Reciprocal Rank Fusion. In Python,
  `db.enable_keyword_search(field="text")` and
  `db.search(query, k, *, text=None, where=None)`, where `query=None` runs a
  keyword-only search. The index is derived data in a new checksummed file,
  `keyword.index`, written atomically by `save()`/`persist()`, caught up from
  metadata after a crash and rebuilt by compaction. `scripts/bench_keyword.py`
  compares it with SQLite FTS5 on a BEIR dataset.
- `SrvDB::rebuild_checksums(path)` / `srvdb.rebuild_checksums(path)`
  recomputes missing or damaged `.crc` checksum files of a closed database
  so it opens again (#33, ADR 0007). It never runs implicitly; an intact
  checksum file that disagrees with the data is refused with
  `ChecksumMismatch` (`RuntimeError` in Python) and nothing is rewritten.

### Changed
- The Python API is a few verbs (#57, ADR 0013):
  `srvdb.create(path, dimension, index="hnsw", *, exist_ok=False, **options)`
  makes a new database (`FileExistsError` if one exists; `exist_ok=True`
  opens it when its recorded settings match, `ValueError` otherwise), with
  the index chosen by name (`flat`, `hnsw`, `ivf`, `sq8`, `pq`, `rabitq`,
  `auto`) and options checked against that index (`m`, `ef_construction`,
  `ef_search`, `nlist`, `nprobe`, `training`, `bits`, `rerank`, `seed`);
  `srvdb.open(path)` reopens it with only the path (`FileNotFoundError` if
  there is none; its unreleased `dimension`/`mode` arguments are removed);
  `db.save()` commits (`persist()` stays as an alias); `db.close()` releases
  the lock without saving; a database is a context manager that saves and
  closes on a clean exit and closes without saving on an exception;
  `srvdb.delete(path)` removes a closed database directory, refusing one
  that holds other files or no SrvDB file. `Database(...)` and the `new_*`
  constructors keep working but are no longer documented. The Python
  reference is retitled around these verbs (`srvdb.create`, `db.add`,
  `db.save`, ...); the per-constructor pages are removed. The LangChain
  adapter's `from_texts` takes `index=` and index options instead of
  `mode=`.
- `db.get(id)` returns `None` for an unknown or deleted id, as the stub
  declared, instead of raising `ValueError`. A negative `k`, `ef_search`,
  `nlist`, `nprobe` or `rerank` raises `ValueError` instead of
  `OverflowError` (#48).
- `create` builds the database in a hidden sibling directory and renames it
  into place, so a failed create leaves nothing behind; `delete` renames the
  locked directory to a hidden tombstone before removing it and refuses
  symbolic links.
- Rust: `SrvDB::create(path, dimension, &CreateOptions)` with `IndexChoice`,
  `SrvDB::delete_database(path)`, `SrvDB::index_name()` and
  `SrvDB::dimension()`; new `Error::NotFound` and `Error::AlreadyExists`.
  `SrvDB::open_existing` on a path without a database returns `NotFound`
  (was `InvalidInput`).
- RaBitQ search uses a fast-scan kernel (ADR 0012). Query tables are
  quantized to u8 and looked up in registers (AVX2 `vpshufb`, NEON `tbl`,
  chosen at runtime, with a scalar fallback) over a block-interleaved
  in-memory copy of the codes. The query rotation is also vectorized.
  Candidate scores before rerank carry a small, bounded extra error from
  the u8 tables; reranked scores are unchanged. The on-disk format is
  unchanged. Resident memory grows by one copy of the codes (#24).

### Added
- Filtered search from Python (#46, ADR 0014): `db.search(query, k,
  where={...})` and `db.search_batch(queries, k, where={...})` return only
  vectors whose metadata matches, and `k` of them whenever at least `k`
  match. Filters are dicts with equality (`{"genre": "jazz"}`), `$eq`,
  `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`,
  `$not` and dotted nested fields; anything else raises `ValueError`. The
  Rust core gains `srvdb::Filter` (`Filter::from_json`, `matches`),
  `SrvDB::search_where` and `SrvDB::search_batch_where`, which validate the
  filter the same way. The LangChain and LlamaIndex adapters pass their
  dict filters, `MetadataFilters`, `node_ids` and `doc_ids` to SrvDB
  instead of post-filtering search results; LangChain dict filters now
  accept the operators above (a dict value is an operator object, a list
  value is rejected; use `$in`), and LlamaIndex `NIN` now matches a
  missing key, like `NE`. A LangChain callable filter, and a LangChain dict
  filter with a dotted key such as `"a.b"` (equality on that literal key,
  as before), still post-filter. Integers compare exactly (2^53 + 1 is not
  2^53). A filter holds at most 1,000 conditions and each `$in` / `$nin`
  list at most 10,000 values; dict keys in `where` must be `str` at every
  level. LlamaIndex queries now raise `ValueError` for: an empty nested
  `MetadataFilters` group (it used to match everything under AND), `IN` /
  `NIN` with a string value (it used to test substrings; pass a list),
  `EQ` / `NE` with a list value (pass `IN`), and more than 10,000
  `node_ids` or `doc_ids`.
- Persisted index settings (#50, ADR 0011): the index mode and its
  parameters (HNSW `m` / `ef_construction` / `ef_search`, IVF `nlist` /
  `nprobe`, RaBitQ bits / rerank / seed, SQ8, PQ, auto) are stored in a
  checksummed `index_settings.bin`, written atomically when a database is
  created and updated by `persist()`. `SrvDB::open_existing(path)` reopens a
  database in its recorded mode with only the path. Opening with a
  conflicting mode, dimension or graph shape fails with the new
  `Error::SettingsMismatch` (Python `ValueError`) and changes nothing.
  Databases from 0.3 and earlier 0.4 builds open as before and record their
  settings on the first open with an explicit mode.
- RaBitQ mode (`SrvDB::new_rabitq`, `RabitqConfig`; Python
  `Database.new_rabitq()` and modes `"rabitq1"`, `"rabitq2"`, `"rabitq4"`):
  training-free 1, 2 or 4-bit codes after a seeded random rotation, with a
  per-vector correction factor, scanned in parallel; the top `k * rerank`
  candidates are rescored exactly against the full vectors, which stay on
  disk as the source of truth. Codes live in `rabitq_codes.bin` (CRC32C
  block checksums) and `rabitq_params.bin` and are rebuilt from the vectors
  when missing, damaged or made with other settings (ADR 0008). The
  benchmark harness gains `rabitq1`, `rabitq2`, `rabitq4` and `--rerank`.
- Python API reference on the documentation site: an index of every public
  object and one page per function, method and adapter class, with
  parameters, return values, exact exceptions, durability and thread-safety
  notes and runnable examples (checked by `website/scripts/check_examples.py`)
  (#42).
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
- Filtered search (Rust): `SrvDB::search_filtered(query, k, &allowed)`
  returns the k nearest neighbours among a set of ids, and
  `SrvDB::ids_where(predicate)` builds that set from metadata. With HNSW
  the filter is applied during graph traversal (non-matching nodes are
  walked through but never returned, with ACORN-style two-hop expansion);
  when scoring the matches directly is cheaper
  (`matches^2 < 4 * ef * M * live`, measured in `docs/benchmarks.md`), or
  in other modes, the matching ids are scored exactly. It always returns k results when at
  least k live vectors match; filtering `search` results afterwards could
  return fewer. The bench harness gains `--filter-pct P`, which compares
  filtered search with post-filtering, and the ignored test
  `filtered_walk_cost` reproduces the threshold measurement. Python
  bindings are unchanged. Design: ADR 0010.

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
- Product-quantized (PQ) similarity scores stay within [-1, 1] (#51). The
  per-subspace cosine assumed unit-length centroids, but k-means reseeds
  empty clusters with raw vectors, so scores up to about 1.09 were seen.
  Both magnitudes are now divided out. Existing codebooks are unchanged;
  scores from them are corrected on the next search.
- `SrvDB::new_scalar_quantized` and `SrvDB::new_quantized` (Python
  `new_scalar_quantized`, `new_product_quantized`) reject a dimension
  outside 128..=4096 with `InvalidInput` (Python `ValueError`) before
  creating anything, matching `DatabaseConfig::new` (#48). Existing
  smaller quantized databases still open with `SrvDB::open_existing`.
  PQ training with a sub-quantizer count that does not divide the
  dimension is `InvalidInput`, and codebooks whose shape is inconsistent
  are rejected on load as `Corrupt` instead of panicking during search.
- `train_ivf()` returns the first error from placing a vector in its IVF
  partition instead of discarding it and reporting success with an
  incomplete index (#49). Its error outside IVF mode names the IVF mode
  rather than a Rust enum variant.
- The `ValueError` for an unknown mode now lists every accepted mode; `ivf`
  and the `scalar` and `product` aliases were missing.
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
