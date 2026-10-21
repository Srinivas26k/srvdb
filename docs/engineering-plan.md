# SrvDB Engineering Plan

Source-level plan for taking SrvDB from v0.2 prototype to a dependable
embedded engine for local AI retrieval and memory. Based on a full read of
the Rust core, Python bindings, tests, CI and benchmarks as of October 2026.

Work is split into three kinds and must be done roughly in this order:

- **Must fix:** correctness, durability, security, licensing.
- **Should build:** features that make SrvDB useful and adoptable.
- **Research:** experiments that may create a real technical advantage.

---

## 0. Current state (baseline)

What already works after the October 2026 fix pass (`fix/priority-fixes`):

- Flat, HNSW, SQ8, PQ and IVF modes all build working storage from both the
  Rust and Python constructors, and reopen correctly.
- Inserts from `add` and `add_batch` reach HNSW and IVF.
- HNSW search returns the correct nearest neighbours (the result-heap
  ordering bug is fixed).
- Python string IDs survive restart; `delete` removes the metadata entry.
- `search` no longer serializes indexes on every call.
- 28 Rust tests pass; builds with and without the `python` feature.

Architecture today:

| Area | Files | Notes |
|---|---|---|
| Facade / routing | `src/lib.rs` | `SrvDB` holds one `Option` per storage and index; routing is if/else chains |
| Vector storage | `src/storage/{flat,sq,pq}.rs` | Append-only file, 24-byte header, mmap reads, count written only on flush |
| Indexes | `src/index/{hnsw,ivf,hybrid}.rs` | In-memory graphs, whole-graph bincode snapshots |
| Metadata | `src/metadata.rs` | `redb`, 1000-entry write cache in front |
| Auto-tuning | `src/core/{strategy,auto_tune}.rs` | Size and RAM thresholds; migrates on `persist()` |
| Python | `src/api/python.rs` | Chroma-style API, string-to-u64 id maps |
| CI | `.github/workflows/release.yml` | Builds wheels on every branch push; publishes on tags. **Runs no tests.** |

---

## 1. Must fix

### 1.1 Durability and recovery (P0)

**Problem.** There is no commit protocol. Concretely:

- `vectors.bin`, `sq8_vectors.bin`, `quantized_vectors.bin`: the vector
  count lives only in the header, rewritten on `flush()`. After a crash the
  file can hold bytes past the recorded count. On reopen the writer seeks to
  end-of-file and appends *after* those orphan bytes, while ids and offsets
  are computed from the stale count, so new vectors are written to the
  wrong place.
- `MetadataStore` keeps up to 1000 writes in an in-memory cache; a crash
  loses them even though `add` returned success.
- Vectors and metadata are written to two stores with no shared
  transaction, so a crash can leave a vector without metadata or the reverse.
- `hnsw.graph` and `ivf/` are whole-file rewrites via `fs::write`. A crash
  mid-write leaves a truncated file; IVF partition write errors are
  silently ignored (`src/storage/ivf.rs`).
- `VectorHeader` is written with native endianness via `unsafe` casts.

**Plan.**

1. Define the guarantee in docs first: *a write is durable once `persist()`
   (renamed or aliased `commit()`) returns; anything after the last commit
   may be lost but never corrupts committed data.*
2. On open, truncate vector files to `HEADER + count * record_size`
   (`storage/flat.rs`, `sq.rs`, `pq.rs`). This alone fixes the misalignment.
3. Make `redb` the commit point: store the committed vector count in a
   `redb` table and update it in the **same** write transaction as that
   batch's metadata. On open, the redb count is authoritative; vector files
   are truncated to it. This gives atomic vector+metadata commits without
   writing a WAL. Evaluate a WAL only if this proves insufficient.
4. Remove the metadata write cache, or flush it inside the commit
   transaction. `redb` batching already gives the throughput.
5. Index files: write to `*.tmp`, `fsync`, atomic rename. Record which
   committed count the index covers; on open, index the missing tail (the
   mechanism `attach_hnsw` already uses) or rebuild if the file is invalid.
   Indexes become derived data: always rebuildable from vector storage.
6. Switch the header to explicit little-endian encoding with a checksum;
   bump `VectorHeader::VERSION` and add a reader for v4 files.
7. **Tests:** a crash harness (`tests/crash/`) that spawns a child process,
   kills it with `SIGKILL` at randomized points during `add`, `persist` and
   index save, then reopens and asserts: committed data intact, no
   uncommitted data visible, search results match an exact-search oracle.

**Acceptance:** the crash harness runs 1,000 randomized kills in CI
without a single corrupted reopen.

### 1.2 Update and delete semantics (P0)

**Problem.** Python `delete` hides the id from results but leaves the vector
in storage and in graph indexes, so deleted vectors still take up the top-k
slots (results come back with fewer than k entries). There is no update,
and the Rust API has no delete.

**Plan.**

1. Add `delete(id)` and `upsert(id, vector, meta)` to `VectorEngine`.
2. Tombstones: a deleted-id bitmap persisted in `redb` and committed
   atomically with the delete.
3. Search skips tombstoned ids *during* traversal (flat scan, HNSW
   candidate expansion, IVF partitions), so results still contain k hits.
4. Upsert = tombstone old internal id + append new vector + remap string id.
5. Compaction: `compact()` rewrites vector files without tombstoned
   records and rebuilds indexes. It is explicit, never automatic, in 0.3.
6. Define deletion guarantees in docs: after `compact()`, deleted vectors
   are physically gone from SrvDB files (does not cover OS/backup copies).

### 1.3 Concurrency semantics (P0)

**Current reality:** `redb` takes a file lock, so a second process opening
the same directory fails. That is accidental, not designed. In-process,
`SrvDB` needs `&mut self` to write, and `HNSWIndex` uses internal locks.

**Plan.**

1. Document: single writer process; many readers in-process; no
   multi-process access.
2. Take an explicit lock file (`srvdb.lock`) on open with a clear error.
3. Python: release the GIL for `add` (currently only search and training
   release it).
4. Remove `unsafe` aliasing where a `MmapMut` is read while the same file
   is written through `BufWriter`; map read-only (`Mmap`) for reads.

### 1.4 Input validation and error handling (P0)

- Reject NaN/inf values in vectors and queries (today they poison
  similarity sorts silently, since `partial_cmp` falls back to `Equal`).
- Reject `k == 0` explicitly; cap k at count.
- Reject invalid JSON metadata instead of silently replacing it with `{}`
  (`src/api/python.rs`, `add`).
- Remove `panic = "abort"` from `[profile.release]`: in a Python extension a
  Rust panic kills the interpreter instead of raising an exception.
- Replace `println!` progress output in library code (PQ training, IVF,
  auto-tuner) with the `log` crate; libraries must not write to stdout.
- Remove remaining `unwrap()` calls on user-reachable paths
  (`partial_cmp(..).unwrap()` in `utils/quantization.rs`).

### 1.5 Smaller correctness bugs (P0/P1)

| Bug | Location | Fix |
|---|---|---|
| `set_ef_search` edits a config copy; the index never sees it | `lib.rs` | Store `ef_search` in `HNSWIndex` behind an atomic; pass through |
| `search_batch` ignores HNSW and IVF | `lib.rs` | Route through `search` per query, in parallel |
| Auto-tuner SQ8 migration does not survive reopen; leaves `vectors.bin` detached | `core/auto_tune.rs` | Persist chosen mode in a manifest; delete or keep raw vectors by explicit policy |
| IVF never reloads: `IVFStorage::load` is unused | `lib.rs` open path | Load on open when `ivf/` exists |
| IVF k-means uses Euclidean, assignment uses cosine | `index/ivf.rs` | Normalize and use one metric |
| Score semantics differ: flat uses `1 - d/2` mapping to [0,1], not raw cosine | `utils/distance.rs` | Return true cosine similarity; document the score per mode |
| `DatabaseConfig` limits dimensions to 128–4096 while header stores u16 | `core/types.rs` | Keep limit but document; allow lower dims (e.g. 64) if tests pass |

### 1.6 CI and packaging (P0)

- Add a `ci.yml` that runs `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test` (with and without `python`), and a Python smoke test on
  Linux, macOS and Windows.
- Release workflow: stop building wheels on every branch push; build on
  PRs and tags only. Gate publishing on the test job.
- Python matrix: drop 3.8 (EOL), add 3.13 and 3.14. Consider `abi3` wheels
  (`pyo3/abi3-py39`) so one wheel covers all versions.
- `pyproject.toml`: remove `numpy`, `psutil` and `scikit-learn` from
  runtime dependencies (benchmark-only; move to an optional `bench` extra);
  replace the placeholder author email.
- Add a `py.typed` stub (`srvdb.pyi`) for editor support.

### 1.7 Licensing (P0)

- **Ownership:** every commit is by Srinivas Nampalli (two git identities,
  same email), so you hold the full copyright and dual licensing needs no
  third-party permission. Before accepting outside contributions, add a CLA
  or DCO-plus-licence-grant policy, otherwise future dual licensing gets
  harder.
- **Inconsistency:** `Cargo.toml` says `AGPL-3.0-only`; the PyPI classifier
  says `AGPLv3+` (or later). Pick one (recommend `AGPL-3.0-only` for a
  dual-licence model) and align both files and the README.
- **Dependencies:** all Rust dependencies are MIT, Apache-2.0, BSD,
  Unlicense or compatible dual licences; none conflict with AGPL or
  proprietary licensing. Add `cargo-deny` to CI to keep it that way.
- Commits before `01c5d83` (2025-12-25 21:44) shipped without a licence
  file; as sole author this does not create obligations, but note it in
  the licence audit.

### 1.8 Honest documentation (P0)

- Remove unverified claims from `src/lib.rs` docs ("100k+ vectors/sec",
  "200+ concurrent QPS") and `src/index/hnsw.rs` (speedup tables).
- Replace README benchmark tables with results from the fixed benchmark
  suite, with hardware listed. Previous HNSW and IVF rows measured flat
  search and must not be reused.
- Document known limitations explicitly (single node, single writer,
  PQ recall on clustered data).

**Release 0.3 = sections 1.1–1.8 complete.** No new index types before then.

---

## 2. Should build

### 2.1 Metadata filtering (0.4, P0)

- Typed metadata: keep JSON for storage, but index declared fields
  (`keyword`, `integer`, `float`, `bool`, `datetime`) in `redb` tables:
  field+value → roaring bitmap of internal ids (`roaring` crate).
- Filter expression API: `eq`, `in`, `range`, `and`, `or`, `not`.
- Execution strategy, chosen by estimated selectivity from bitmap
  cardinality:
  - very selective (< ~1–5% of ids): exact scan over the bitmap;
  - otherwise: filtered traversal, where HNSW and IVF skip non-matching ids
    during candidate expansion (same hook as tombstones in 1.2) with
    `ef` increased adaptively.
- Never post-filter a fixed top-k.
- **Benchmark:** filtered recall@10 versus exact filtered search at
  selectivities 0.1%, 1%, 10% and 50%.

### 2.2 Hybrid retrieval (0.4, P0)

- Lexical index: evaluate `tantivy` (mature, MIT) as an optional feature
  versus a small built-in BM25 inverted index in `redb`. Prefer the built-in
  index if it keeps the core small and meets quality targets.
- Fusion: Reciprocal Rank Fusion first (no tuning required); weighted
  score fusion as an option.
- API: `search(query_vector=..., query_text=..., filter=..., k=...)`.

### 2.3 Reranking and quantized-then-exact search (0.4, P1)

- When raw vectors are kept, quantized modes fetch `k * oversample`
  candidates and rerank with full-precision cosine. Today SQ8 and PQ return
  quantized scores directly.
- Make "keep raw vectors on disk" an explicit storage option, so memory
  stays small (codes in RAM) while reranking reads mmapped raw vectors.

### 2.4 Performance fixes with known causes (0.4, P1)

| Issue | Cause | Fix |
|---|---|---|
| SQ8 search ~45 ms | Decodes and allocates a full f32 vector per candidate; single-threaded; full sort | Precompute query-side scale and offset; integer dot product on u8 with SIMD; rayon batches; `select_top_k` |
| PQ recall collapses on clustered data | ADC averages per-subspace cosines against normalized centroids, discarding magnitudes | Normalize the full vector once, train un-normalized centroids, sum inner products in the lookup table |
| HNSW memory per node | `HashMap<u64, Node>` with nested `Vec`s | Dense `Vec` indexed by internal id; flat neighbour arrays |
| HNSW recall on clustered data | `select_neighbors_simple` | Implement the paper's heuristic neighbour selection (Algorithm 4) |
| HNSW build is single-threaded | Sequential inserts | Parallel insertion with per-node locks |

### 2.5 Observability (0.4, P1)

- `db.stats()`: counts, tombstones, bytes per vector, index coverage,
  memory estimate per component.
- `search(..., explain=True)`: strategy used, candidates visited, filter
  selectivity, timing per stage.

### 2.6 Developer experience (0.3–0.4)

- Done in 0.3: `SrvDBPython` renamed to `srvdb.Database` (old name kept as an alias for
  one release); accept numpy arrays directly (zero-copy via `numpy` crate)
  instead of `list[list[float]]`.
- Examples: local RAG, offline document search, filtered multi-tenant
  search, low-memory device configuration.
- `CONTRIBUTING.md` update, `SECURITY.md`, issue templates, good first
  issues taken from section 1.5.

### 2.7 Later releases (summary)

- **0.5, resource-aware retrieval:** see research 3.2; ships only after the
  study shows the recommendations beat a fixed default.
- **0.6, agent memory:** timestamps as a first-class indexed field (builds on
  2.1), namespaces (one bitmap per namespace, or separate directories),
  TTL/expiry as explicit tombstoning jobs, append-only change log for
  provenance. A separate `srvdb.memory` module layered on the core; the core
  takes no agent dependencies.
- **0.7, portability:** stable Rust API (`#[non_exhaustive]` configs,
  semver checks via `cargo-semver-checks`), then a C API via `cbindgen` if
  users ask for it. WebAssembly and mobile are separate investigations.

---

## 3. Research

Each item is a time-boxed experiment with a written result, published
whether positive or negative.

### 3.1 Quantization study

- **Question:** at equal recall@10, which representation gives the lowest
  memory and latency on SrvDB's target hardware?
- **Candidates:** f32 exact; f16; SQ8 (fixed); PQ (fixed); binary
  (1-bit sign) with rerank; TurboQuant-style rotation plus low-bit scalar
  codes.
- **TurboQuant / TurboVec:** read the Google Research paper and implement
  from the paper, or integrate TurboVec only after checking its licence for
  compatibility with AGPL and with your commercial licence (a copyleft or
  restrictive licence would block the commercial edition). Benchmark
  TurboVec itself as a competitor either way.
- **Datasets:** SIFT-1M or GloVe (standard ANN benchmarks), plus a
  real-embedding set (e.g. MiniLM or BGE embeddings of a Wikipedia subset)
  because random data hides clustering problems.
- **Metrics:** recall@10/@100, p50/p99 latency, bytes/vector, peak RSS,
  build time, insert latency.

### 3.2 Accuracy-aware resource planning

- Replace the fixed thresholds in `core/strategy.rs` with a planner:
  inputs = memory budget, recall target, dataset size and dimension,
  measured CPU features; output = representation + index + parameters,
  with an explanation.
- Calibrate with measured cost models from 3.1, not guesses.
- Deterministic and explainable; recommend by default, switch
  automatically only when the user opts in.
- **Success test:** across a grid of budgets and datasets, the planner's
  choice is within 10% of the best measured configuration.

### 3.3 Incremental compressed indexing

- Can codes be appended and indexed incrementally without retraining?
  Measure recall drift as data distribution shifts after training;
  define a retrain trigger.

### 3.4 Filtered ANN

- Compare filtered HNSW traversal against approaches from the literature
  (e.g. ACORN-style predicate-aware graphs) under selective filters.

Novelty is not claimed for any of this until a prior-art review is done.

---

## 4. Benchmark methodology (applies to everything above)

The current `universal_benchmark.py` measures single-process Python
end-to-end latency on synthetic data. Keep it as a smoke benchmark, and add
a rigorous suite:

- Rust-level harness (`criterion` plus a standalone runner) to separate
  engine time from Python overhead.
- Standard datasets with published ground truth (ann-benchmarks format).
- Recall versus QPS curves, not single points; report at fixed recall
  targets.
- Competitors run at their documented recommended settings: FAISS,
  hnswlib, LanceDB, sqlite-vec, TurboVec.
- Record hardware, OS, versions and commit hash in every result file; check
  results into a `benchmarks/results/` directory.

---

## 5. Sequencing (first 90 days)

| Days | Work | Exit criterion |
|---|---|---|
| 1–15 | 1.6 CI with tests; 1.4 validation and `panic=abort`; 1.7 licence alignment; 1.8 doc cleanup; 1.5 small bugs | CI green on 3 OSes; README numbers reproducible |
| 16–30 | 1.1 durability; 1.2 delete and upsert; 1.3 concurrency docs and lock | Crash harness passes; tag **0.3.0** |
| 31–60 | 2.4 SQ8 and PQ fixes; 2.3 rerank; 2.1 filtering; 3.1 quantization study | Filtered recall benchmark published |
| 61–90 | 2.2 hybrid search; 2.5 observability; 2.6 examples; benchmark report and articles | **0.4.0** released; external users testing |

---

## 6. Explicitly out of scope until 1.0

Distributed operation, replication, GPU acceleration, a hosted service, a
dashboard, and new index families beyond those listed here.
