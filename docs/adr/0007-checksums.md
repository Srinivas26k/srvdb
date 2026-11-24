# ADR 0007: CRC32C checksums for vector and index files

- **Status:** Accepted
- **Date:** 2026-11-23

## Context

ADR 0003 makes committed data survive crashes, but nothing detected damage
that happens after a write: bit rot, a torn sector, a bad copy or restore.
Index and quantizer files carried a CRC32 in their envelope, but the vector
files (`vectors.bin`, `sq8_vectors.bin`, `quantized_vectors.bin`), the only
source of truth for vectors, had none. A flipped bit there was read as a
valid vector and returned in search results.

Constraints:

- Vector files are appended and read through a memory map; search scans
  record slices with zero copies. Records must stay contiguous.
- ADR 0003's guarantee and commit protocol must not change.
- Databases written by SrvDB 0.3 must keep opening.
- AGENTS.md: use an established checksum crate, never hand-rolled code.

## Decision

### Algorithm and crate

CRC32C (Castagnoli), via the `crc32c` crate (MIT OR Apache-2.0, maintained,
widely used). It uses the SSE4.2 / ARMv8 CRC instructions when available
and a table fallback otherwise, and provides `crc32c_append` to extend a
checksum without rereading earlier bytes. CRC32C detects all single-bit and
burst errors up to 32 bits and is the checksum storage systems commonly use
for this purpose (ext4 metadata, iSCSI, RocksDB). It is not a defence
against deliberate tampering; nor is it meant to be.

### Granularity

- **Vector files: one CRC32C per 1 MiB block** of the record area (after
  the 24-byte header). The table is stored in a sidecar `<file>.crc`
  (for example `vectors.bin.crc`) inside the existing checksummed envelope
  (`storage/format.rs`, kind `CRCS`). The last block is partial.
  Per block, not per file, because appends only change the last block: the
  in-memory table is updated incrementally, and a failure names the damaged
  byte range. Per block, not per record, because an inline per-record CRC
  would break the contiguous zero-copy layout that flat and SQ8 search
  scan, and per-record tables cost 4 bytes per vector instead of 4 bytes per
  MiB. A sidecar, not a table inside the vector file, because the file
  grows at the end and has no room for a growing table; the sidecar is
  replaced atomically (temp file, fsync, rename, directory fsync).
- **Index and quantizer files** (`hnsw.graph`, `ivf.index`,
  `scalar_quantizer.bin`, `codebooks.bin`): one CRC32C per file, in the
  envelope. They are always written and read whole, so a finer granularity
  buys nothing.

### Ordering and crash behaviour

The sidecar covers exactly `count` records, and that count is the file's
record count on open (the header count is informational for version 5).

- `RecordFile::sync` (step 1 of `persist()`): write the header, fsync the
  vector file, then replace the sidecar. The sidecar never covers bytes
  that are not durable. A crash before the sidecar rename leaves the old
  sidecar, whose count is still at least the committed count, so recovery
  truncates as before.
- `truncate_to` (recovery): save the sidecar for the shorter length first,
  then shrink the file. A crash in between reopens with the new count and
  truncates again.
- Creation: the empty sidecar is written before the vector file, so a
  version 5 header always has a sidecar.
- Compaction: the staged `<file>.compact` gets its own
  `<file>.compact.crc`, written and fsynced before the redb commit;
  `compaction::finish` renames both and `discard_abandoned` removes both.
  `finish` completes before any vector file is opened.

redb remains the only commit point. A missing sidecar next to a version 5
file is `Error::Corrupt`.

### When checksums are verified, and the cost

- **On open**: every block of every vector file is verified before the
  database is usable (before recovery truncates). Cost: one sequential read
  of each vector file at open, plus the CRC computation, which with
  hardware CRC instructions is far cheaper than the read. Open time
  therefore grows with data size.
- **Not on each read**: after open, searches read the memory map without
  re-verifying, so search speed is unchanged. Damage that appears while a
  database is open is detected at the next open.
- **Index and quantizer files**: verified whenever they are loaded (on open).
- **Per `persist()`**: computing CRCs of appended bytes as they are written,
  plus one atomic sidecar replacement per vector file (two extra fsyncs and
  4 bytes per MiB of data written).

No performance numbers are claimed here; measure with the benchmark
harness before quoting any.

### Errors

A new variant, `Error::ChecksumMismatch`, means data does not match its
checksum. The message names the file and, for vector files, the byte range,
and says to restore from a backup (an index file can be deleted instead; it
is rebuilt). A damaged `hnsw.graph` is already rebuilt from the verified
vectors on open, as before. Structural problems (truncation, bad magic or
version, a sidecar that does not match the file) stay `Error::Corrupt`.
Python maps both to `RuntimeError`.

### Format versions and migration

- Vector file header version 4 -> 5 (5 = sidecar required). A version 4
  file (SrvDB 0.3) is read as before, its checksums are computed from the
  data as found, saved, and then the header is set to version 5. A crash
  between the two leaves a version 4 file that upgrades again. Data in a
  version 4 file cannot be verified retroactively.
- Envelope version 1 -> 2: version 2 uses CRC32C; version 1 (CRC32 IEEE,
  via `crc32fast`) is still read, and files are rewritten as version 2 the
  next time they are saved. `crc32fast` stays as a dependency for that
  reader.
- Fixtures written by SrvDB 0.3 live in `crates/srvdb/tests/data/v0.3`
  and are opened by `tests/test_checksums.rs` and the `format` unit tests.

### Recovering a missing or damaged sidecar (amendment, issue #33)

A missing or damaged `<file>.crc` makes open fail even when the records
are intact (for example after copying only `vectors.bin`). Open never
repairs this by itself: a sidecar rebuilt silently from the data would
bless any damage in it. Recovery is an explicit call,
`SrvDB::rebuild_checksums(path)` (Python `srvdb.rebuild_checksums`), on a
closed database (it takes the lock). For each vector file with a version
5 header:

- **Intact sidecar** (envelope CRC holds, record size and block count
  match) **that matches the records**: left alone.
- **Intact sidecar that does not match the records** (or claims more
  records than the file holds): the data is damaged, not the sidecar. The
  call fails with `ChecksumMismatch` (`Corrupt` for truncation) and writes
  nothing; the user is told to restore from a backup.
- **Missing sidecar, or one that fails its own envelope check or does not
  describe the file**: recomputed from the records as found. This is the
  only case that trusts the data, and only because the user asked; the
  documentation says a rebuilt checksum cannot detect damage that was
  already there. Deleting a sidecar and rebuilding is the documented way
  to force that trust.

The new table covers `min(header count, whole records present)`, the rule
of the version 4 upgrade. The header count is written and fsynced before
the redb commit, so it is never below the committed count; open then
truncates to the committed count as usual. The sidecar is written with
`write_atomic` (temp file, fsync, rename, directory fsync): a crash leaves
the old or the new sidecar and the call can be repeated. Vector files,
redb and indexes are not modified. The record size comes from the header
(flat, SQ8, RaBitQ) or from `codebooks.bin` (PQ). Staged compaction files
(`*.compact.crc`) are not rebuilt; `compaction::finish` handles them on
open.

## Alternatives considered

- **Store block CRCs in redb, in the commit transaction**: atomic with the
  committed count, but couples every vector file to redb, complicates
  compaction and the storage-level types (`VectorStorage` is usable without
  redb), and would make redb hold derived data about another file.
- **Per-record CRC inline in the vector file**: breaks contiguous records
  (zero-copy batch reads and SQ8 scans) and costs 4 bytes per vector.
- **One CRC per vector file**: every append would need the whole file
  reread or a growing CRC, and a failure could not be localised.
- **Verify on every read**: puts a CRC on the search hot path for every
  scanned record; damage during one open session is rare enough that open
  time verification is the better trade-off. Can be revisited behind an
  option if needed.
- **`crc-fast`**: also suitable; `crc32c` is smaller and has the append
  API we need.

## Consequences

- Silent damage in committed vectors becomes a typed open error instead of
  wrong search results.
- Open reads every vector file once; very large databases open more slowly.
- Each vector file has a sidecar that must be copied with it; backups must
  include `*.crc` files.
- SrvDB 0.3 cannot open a database after 0.4 has upgraded it (header
  version 5 is rejected as newer).
