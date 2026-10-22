# ADR 0005: Concurrency contract

- **Status:** Accepted
- **Date:** 2026-10-22

## Context

SrvDB assumes a single writer: the commit protocol (ADR 0003), recovery
on open and compaction (ADR 0004) all rewrite files that a second
writer would corrupt. Until now nothing enforced this. Two processes
opening the same directory were only stopped by redb's own lock on
`metadata.db`, which surfaced as an internal error, and thread-safety
was not documented.

## Decision

### Processes: one open instance per database

Every constructor first takes an exclusive OS lock (`flock` on Unix,
`LockFileEx` on Windows, via `std::fs::File::try_lock`) on `srvdb.lock`
in the database directory, before any other file is opened, and holds
it until the `SrvDB` is dropped. A second open, in the same or another
process, fails immediately with `Error::Locked` (Python:
`BlockingIOError`). The OS releases the lock when a process exits, even
on a crash, so a stale lock never blocks recovery.

### Threads: shared reads, exclusive writes

- `SrvDB` is `Send + Sync` (checked at compile time by a test).
- Searches take `&self`, so many threads can search one `Arc<SrvDB>`
  at once.
- Writes take `&mut self` and are exclusive. Rust's borrow rules enforce
  this; applications mixing readers and a writer wrap the database in an
  `RwLock`.

### Python

- A `Database` is safe to share between threads without any lock of
  the caller's. Its state (the `SrvDB` and the string-id maps) sits
  behind one `RwLock`: searches take it shared and run in parallel,
  writes take it exclusively, and other calls wait instead of failing.
- Every method releases the GIL before waiting for the lock. A thread
  that held the GIL while waiting could block a writer that needs the
  GIL, and both would hang. Python objects are built only after the lock
  is released.
- A panic while the lock is held poisons it; later calls raise a
  `RuntimeError` asking to reopen the database instead of using
  half-updated state.
- (Revised: an earlier version raised an error when a call overlapped a
  write and left locking to the application.)

## Alternatives considered

- **Lock inside SrvDB (`RwLock` around all state):** hides contention and
  costs every single-threaded user; Rust's `&`/`&mut` already expresses
  the rule at no cost.
- **Multi-process writers:** would need a shared write-ahead log and
  coordination; out of scope for an embedded single-node database.
- **A lock crate (fs2, fd-lock):** unnecessary since Rust 1.89 provides
  file locking in the standard library; the workspace now declares
  `rust-version = "1.89"`.

## Consequences

- Opening a database a second time while it is open fails fast with a
  clear error instead of risking corruption.
- Network file systems with unreliable advisory locking are not
  supported for concurrent access.
