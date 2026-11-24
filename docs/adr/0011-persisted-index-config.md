# ADR 0011: Persisted index settings in `index_settings.bin`

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Until now the index mode was not stored. A database created as HNSW, SQ8,
PQ, IVF or RaBitQ reopened as flat unless the caller passed the mode again
(only an existing `hnsw.graph` or `ivf.index` was picked up implicitly).
Issue #50 asks for the mode and its parameters to be stored, so issue #57
can offer `srvdb.open(path)` with only a path.

Constraints:

- ADR 0003: redb's commit transaction is the only commit point; vector
  files are the source of truth and indexes are derived data.
- ADR 0007: every SrvDB file is checksummed (CRC32C) and replaced
  atomically.
- Databases from SrvDB 0.3 and earlier 0.4 builds have no settings and must
  keep opening as they did.
- AGENTS.md section 5: validate everything read on open; typed errors; no
  silent rewriting of documented behaviour.

## Decision

### Where: a small manifest file

The settings live in `index_settings.bin` in the database directory, in the
existing checksummed envelope (`storage/format.rs`, kind `IDXS`, envelope
version 2 = CRC32C), written with `write_atomic` (temp file in the same
directory, fsync, rename, directory fsync).

- **Not the vector header.** There are three vector files (one per storage
  kind), the 24-byte header has 6 reserved bytes, and RaBitQ, IVF and HNSW
  parameters do not fit. Growing the header is a vector-file format change
  for every database, for data that is not about vectors.
- **Not redb.** redb would make the write transactional, but the settings
  are needed to decide which storage to open before metadata is touched,
  and a corrupt settings record must be reportable without opening the
  commit store. A manifest also matches every other non-vector file
  (`hnsw.graph`, `codebooks.bin`, `rabitq_params.bin`): one envelope, one
  reader, one set of corruption tests.

### What is stored

The payload (postcard, little-endian varints) is a fixed record with its
own `format` number (1): `kind`, `dimension`, HNSW `m`, `ef_construction`,
`ef_search`, IVF `nlist`, `nprobe`, RaBitQ `bits`, `rerank`, `seed`. Kinds:
flat, hnsw, sq8, pq, hnsw+pq, ivf (with the partition HNSW parameters),
rabitq, auto. Fields a kind does not use must be zero. PQ parameters are
not duplicated: the trained `codebooks.bin` already holds them, checksummed.

Validation on load: envelope magic, kind, version and CRC32C (damage is
`ChecksumMismatch`, anything else `Corrupt`); payload format; known kind
(an unknown one is `Corrupt`, "written by a newer SrvDB? upgrade SrvDB");
every parameter in range (`nprobe <= nlist`, RaBitQ settings through
`RabitqConfig::validate`); no stray fields.

### When it is written, and the guarantee

- **Constructors.** A new database, or a database without settings opened
  with an explicit mode (`new_with_hnsw`, `new_scalar_quantized`,
  `new_rabitq`, `new_with_config` naming a mode, ...), has its settings
  written before the constructor returns. After it returns, the settings are
  durable.
- **`persist()`.** After the commit point and the derived indexes, the
  settings are rewritten if they changed (`set_mode`, `configure_ivf`,
  `set_ef_search`, `set_rabitq_rerank`, auto mode). `persist()` never
  creates the file: a legacy database opened by default stays unpinned.
- **Crash.** The rename is atomic: a crash leaves the old or the new file.
  Losing the newest settings loses only a mode change since the last
  `persist()`; committed vectors and metadata are unaffected (ADR 0003
  unchanged). A crash before the first write leaves a database that looks
  like a 0.3 one and is handled as below.

### Opening

- `SrvDB::open_existing(path)` reads the settings and opens the database in
  that mode with those parameters. A database without settings opens flat if
  it has `vectors.bin` (dimension from its header, as before); a quantized
  one returns `InvalidState` asking for one explicit open.
- `new_with_config` with a default configuration (flat, no quantization)
  behaves like `open_existing` for an existing database with settings.
- **Conflicts.** A constructor asking for another mode, another dimension or
  another graph shape (HNSW `m` / `ef_construction`, IVF `nlist`) fails with
  the new `Error::SettingsMismatch` (or `DimensionMismatch`) before anything
  is written. Search-time knobs (`ef_search`, `nprobe`, `rerank`) are not
  conflicts: the caller's value is used and recorded by the next
  `persist()`. RaBitQ bit width and seed are also not conflicts: ADR 0008
  already rebuilds the codes from the vectors for other settings.

### Migration

0.3 and early 0.4 databases have no file and open exactly as before. The
first explicit-mode open records the settings (test with the committed
`tests/data/v0.3/hnsw` fixture). No existing file format changes, so no
version bump of any existing file.

## Alternatives considered

- **Silently adopt the stored settings and ignore the caller's.** Hides
  mistakes such as opening an HNSW database as SQ8; rejected for a typed
  error.
- **Store the settings in each index file.** `hnsw.graph` already holds its
  config, but flat, SQ8 and auto modes have no index file, and derived files
  may be deleted to force a rebuild.
- **JSON manifest.** Human-readable, but would need a second parser and
  checksum scheme next to the envelope.

## Consequences

- Reopening needs only the path; #57 can build on `open_existing`.
- `Error` gains `SettingsMismatch` (the enum is `#[non_exhaustive]`);
  Python maps it to `ValueError`.
- Behaviour change: reopening HNSW with a different `m` / `ef_construction`
  used to keep the stored graph silently; it is now an error. Reopening a
  database in another mode is now an error instead of opening wrong or
  empty storage.
- One more file to back up with the directory.
