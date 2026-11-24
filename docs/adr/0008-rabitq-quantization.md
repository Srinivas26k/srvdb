# ADR 0008: RaBitQ-style low-bit quantization with rerank

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

SrvDB 0.4 targets laptops and edge devices (roadmap items B1 and B3,
issue #23). The existing compressed modes both need training and both
throw the full vectors away: SQ8 learns per-dimension ranges (4x smaller),
PQ learns k-means codebooks (more compression, recall bounded by the
codebooks, training cost grows with data). Recent work shows that a random
rotation followed by a few bits per dimension, with a per-vector
correction factor, gives distance estimates with provable error bounds and
no training at all:

- J. Gao, C. Long. *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search.*
  SIGMOD 2024. arXiv:2405.12497.
- J. Gao, Y. Gou, Y. Xu, Y. Yang, C. Long, R. C.-W. Wong. *Practical and
  Asymptotically Optimal Quantization of High-Dimensional Vectors in
  Euclidean Space for Approximate Nearest Neighbor Search* (extended
  RaBitQ, multi-bit codes). SIGMOD 2025. arXiv:2409.09913.
- A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni. *TurboQuant: Online
  Vector Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874
  (2025); the `turbovec` library implements it.

Constraints: indexes are derived data and must be rebuildable from vector
storage (AGENTS.md section 4); every on-disk structure is versioned,
checksummed (ADR 0007) and validated on open; no new dependencies are
needed; hand-written SIMD is milestone M4 (issue #24), so this ADR must fix
a code layout that a fast-scan kernel can use without a format change.

## Decision

### A new mode, not a new index structure

`SrvDB::new_rabitq(path, dimension, RabitqConfig)` opens a database that
keeps every vector at full precision in `vectors.bin` (unchanged, the
source of truth) and additionally keeps RaBitQ codes. Searches scan the
codes (exhaustively, like flat and SQ8, parallel over 256-record blocks on
rayon) and rerank. It is a compressed scan, not a new graph or partition
structure. `RabitqConfig` is `#[non_exhaustive]`; `bits` is validated
against the fixed set {1, 2, 4}.

### Quantizer

For a vector `x` and `b` bits per dimension:

1. `y = R x / |x|` where `R` is a random orthogonal matrix. `R` is
   generated from a 64-bit seed: SplitMix64 + Box-Muller Gaussians,
   orthonormalised with two passes of modified Gram-Schmidt in f64. Our
   own generator, not `rand`, so a dependency upgrade can never change
   the matrix a seed produces.
2. Codes `c_i in [0, 2^b)` with grid values `g_i = c_i + 1/2 - 2^(b-1)`
   approximate `t * y_i`. The scale `t` is chosen per vector from 32
   candidates to maximise `<g, y> / |g|` (extended RaBitQ searches the
   same objective exactly; a coarse search is within a fraction of a
   percent, see `grid_alignment_matches_theory`). For `b = 1` the codes
   are the signs, exactly RaBitQ.
3. One f32 correction factor `f = 1 / <g, y>` is stored per vector.

For a query `q`, `q' = R q / |q|`, and the cosine similarity is estimated
as `f * <g, q'>`, which equals the RaBitQ estimator
`<o_bar, q> / <o_bar, o>` with `o_bar = g / |g|`. Under the random
rotation it is unbiased, with error `O(1/sqrt(D))` scaled by how far
`<o_bar, o>` is from 1 (about 0.80 for 1 bit, 0.94 for 2, 0.995 for 4).
SrvDB's metric is cosine, so the vector norm is not needed and the data is
not centred: no training pass, no centroid. The query stays in f32 (RaBitQ
also quantizes it; that is an M4 speed optimisation, not a correctness
need).

### Dimension limit

`new_rabitq` refuses dimensions above 4096 (`Error::InvalidInput`), the
same maximum as `DatabaseConfig` and `srvdb.open` (128 to 4096). The
rotation is a dense `dim x dim` matrix, built in f64 and kept in f32, and
regenerated with O(dim^3) work on every open: about 200 MB and seconds at
4096, but some 51 GB at the 65535 the vector-file header could express,
which would abort or hang the host. A structured (Hadamard) rotation that
lifts this limit is milestone M4 work.

### Rerank

The scan keeps `k * r` candidates by estimated similarity, then rescores
them exactly against `vectors.bin` and returns the top `k`. Defaults are
`r = 40, 20, 10` for 1, 2 and 4 bits (fewer bits, more candidates), from
the recall tests on clustered data; `r = 0` returns estimates without
reranking. `r` is a search-time setting (`set_rabitq_rerank`), not stored.

### Files and validation

- `rabitq_params.bin`: postcard payload `{dimension, bits, seed,
  rotation fingerprint}` (format version 1) in the existing envelope,
  kind `RBQP`, CRC32C, written atomically. The fingerprint is the CRC32C
  of the generated matrix: if a seed ever expands differently (code change,
  platform `ln`/`cos` differences) the codes are detected as stale.
- `rabitq_codes.bin`: a `RecordFile` (24-byte versioned header, records
  at fixed offsets, `.crc` sidecar of 1 MiB CRC32C blocks, ADR 0007). The
  header mode byte is `0x10 | bits`, so a file of another width is refused.

On open, a mismatch (other bit width, seed, dimension or fingerprint, or a
code-file header of another width), damage (`Corrupt`,
`ChecksumMismatch`) or a missing file causes the codes to be deleted and
rebuilt from `vectors.bin`, with a log record; the rebuilt codes are
fsynced at once, so a read-only session does not re-encode on every open.
Any other error (I/O) is returned: it does not prove the codes are bad. Codes are
fsynced after the commit point (step 3 of ADR 0003, with the other derived
indexes); on open they are truncated or extended to the committed vector
count. Compaction deletes them and they are rebuilt for the new ids.

### Code layout

One record per vector: the f32 factor (little-endian), then the codes
packed `8 / b` dimensions per byte, lowest bits first, dimensions in order
(`ceil(D * b / 8)` bytes; 4 + 48, 4 + 96 and 4 + 192 bytes at `D = 384`).

The scalar kernel builds one 256-entry table per code byte (the partial
dot product of every possible byte with the query), so all bit widths
share one loop: `f * sum_j LUT_j[byte_j]`. This is the layout fast-scan
kernels consume: for `b = 4` each nibble is one dimension, for `b = 1`
each nibble is four dimensions, and a `pshufb` / `tbl` kernel uses
16-entry nibble tables over the same bytes. Fast-scan wants the bytes of
32 vectors interleaved; because the codes are derived data that are
rebuilt freely, M4 can build an interleaved in-memory copy on open, or
introduce a block-interleaved record file as `rabitq_codes.bin` version 2
with the normal rebuild path, without touching `vectors.bin`. We chose
per-vector records now because they reuse `RecordFile` (append, truncate,
checksums, crash rules) unchanged.

**Update (ADR 0011):** the 256-entry tables are replaced by the
in-memory option. Codes are copied into 32-vector interleaved blocks on
open and scanned with u8 nibble tables (AVX2 `vpshufb`, NEON `tbl`, and
a scalar fallback). `rabitq_codes.bin` is unchanged.

### Build-time measurement

Encoding runs on the rayon pool (commit a9d7991). Before that change,
building at dim 768, n 50 000 took 44 s (1 bit) and 57 s (4 bits); after
it, 14 s and 20 s; flat took 1.4 s. Environment: AMD Ryzen 5 3550H (8
logical cores), Ubuntu 26.04.1 LTS, kernel 7.0.0-43-generic, release
build; dataset synthetic-gauss (dim 768, n 50 000, 500 clusters, seed 42,
200 queries); "before" at commit 3d2f17f and "after" at a9d7991, both
with the bench-mode changes not yet committed (later 2a42541);
command `cargo run --release -p srvdb --example bench -- --dim 768
--n 50000 --modes flat,sq8,rabitq1,rabitq2,rabitq4` (the after run used
`--modes rabitq1,rabitq4`). A later full run at merge commit e3c6971 gave
13.1 s, 20.1 s and 17.7 s for 1, 2 and 4 bits.

## Alternatives considered

- **Store only the codes (drop f32 vectors).** Smallest footprint, but
  the index would become the only copy of the data (forbidden) and rerank
  impossible. A later "codes in memory, vectors on disk" setting already
  falls out of this design: the f32 file is memory-mapped and only `k * r`
  rows are touched per query.
- **Fast Walsh-Hadamard rotation.** O(D log D) instead of O(D^2) per
  vector, but needs a power-of-two dimension (padding 384 to 512 costs a
  third more code bits). The dense rotation costs O(D^2) per insert and
  query, small next to an O(n D) scan; revisit with a benchmark.
- **Centre on a data mean (IVF-RaBitQ).** Better estimates for data with a
  large common offset, but it is a training pass and the centroid becomes
  state to keep consistent; left for a later ADR with measurements.
- **Store the rotation matrix.** D^2 floats (2.3 MB at D = 768) for no
  gain over seed + fingerprint.
- **Reuse SQ8 with fewer bits.** SQ8's ranges are trained per dimension
  and its estimate is biased at low bit widths; the rotation and
  correction factor are what make 1-2 bits usable.

## Consequences

- RaBitQ databases are larger on disk than flat ones (f32 plus codes) but
  scan far fewer bytes per query; the win is memory bandwidth and resident
  set, not disk.
- Inserts pay one O(D^2) rotation per vector; opening pays O(D^3) once to
  generate the rotation (well under a second at D = 768 in release).
- The rerank factor trades latency for recall and its best value depends
  on the data; the defaults are documented, not tuned per dataset.
- We now maintain the rotation generator: changing it changes every
  fingerprint, which triggers a rebuild of every RaBitQ database on open.
  That is safe but slow, so it must be called out in release notes.
- HNSW and IVF do not use the codes yet; combining them is future work.
