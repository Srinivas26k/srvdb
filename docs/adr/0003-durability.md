# ADR 0003: Durability guarantee and commit protocol

- **Status:** Accepted
- **Date:** 2026-10-21

## Context

Through v0.2, SrvDB had no defined commit point:

- The vector count lived only in the header of `vectors.bin` (and the SQ8
  and PQ equivalents) and was rewritten on flush. After a crash the file
  could hold bytes past the recorded count. Reopening appended after those
  bytes while computing offsets from the stale count, so new ids pointed
  at the wrong vectors.
- Metadata went to `redb` through a 1,000-entry in-memory cache that was
  lost on a crash, so vectors and metadata could disagree.
- HNSW and IVF index files were rewritten in place; a crash mid-write left
  a truncated file, and IVF write errors were ignored.

## Decision

### The guarantee

> When `persist()` returns successfully, every earlier `add`, `add_batch`
> and `delete` survives a process crash or power loss. After a crash the
> database reopens in exactly the state of the last successful `persist()`:
> later writes are absent, never partially present, and nothing committed
> is corrupted.

Writes after the last `persist()` may be lost. Searches see writes as soon
as the call that made them returns, before `persist()`.

### Commit protocol

`redb` is the single source of truth for what is committed. It is already
crash-safe and transactional, so SrvDB does not add its own write-ahead log.

1. `add` / `add_batch` append vectors to the vector file and write metadata
   in a `redb` transaction with `Durability::None` (cheap, no fsync; it
   becomes durable only with a later durable commit).
2. `persist()`:
   1. flushes and fsyncs the vector file,
   2. commits one `redb` transaction with `Durability::Immediate` that
      records `committed_count` (and, later, tombstones),
   3. saves index files atomically (below).
3. Opening a database:
   1. reads `committed_count` from `redb` (for a v0.2 database without it,
      takes the vector-file header count and records it),
   2. truncates the vector file to exactly `committed_count` records,
      discarding any uncommitted tail,
   3. deletes metadata for ids at or above `committed_count`.

Because `redb` rolls back non-durable commits on a crash, metadata and the
committed count always agree with the truncated vector file.

### Index files

HNSW and IVF files are derived data. They are written to a temporary file,
fsynced, and renamed over the old file, so a reader sees the old or the new
version, never a mix. On open, a missing or unreadable index is rebuilt from
the vectors, and vectors beyond what an index covers are inserted.

### File formats

Each file SrvDB writes carries a magic number, a format version and a
checksum, and is validated on open; a mismatch is a `Corrupt` error.
`bincode` (unmaintained) is replaced. A small reader keeps v0.2 quantizer
files (`scalar_quantizer.bin`, `codebooks.bin`) readable; v0.2 index files
are rebuilt instead of read.

### redb upgrades

`metadata.db` is a redb database. SrvDB moved from redb 2 to redb 4,
whose file format (v3) cannot read v2 files. Opening a v2 `metadata.db`
converts it once with redb 2.6's `Database::upgrade()` (kept as a
migration-only dependency, `redb2`), then opens it with redb 4.

## Alternatives considered

- **Own write-ahead log:** duplicates what `redb` already provides and adds
  recovery code to maintain.
- **Store vectors inside `redb`:** loses zero-copy memory-mapped reads, the
  main reason SrvDB's flat search is fast.
- **fsync on every add:** simple but makes ingestion orders of magnitude
  slower; `persist()` gives callers control of the trade-off.

## Consequences

- Callers must call `persist()` for durability; documented on the API and in
  the README.
- A crash test (kill a child process at random points, reopen, verify) is
  required for every change to the write path.
- Opening a database may truncate files; this only removes data that was
  never committed.
