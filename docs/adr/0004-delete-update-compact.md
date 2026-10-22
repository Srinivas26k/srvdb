# ADR 0004: Delete, update and compaction

- **Status:** Accepted
- **Date:** 2026-10-22

## Context

Through v0.2, Python `delete()` only removed an id from the in-memory
map. The vector stayed in storage and in every index, so deleted vectors
kept occupying top-k slots (searches returned fewer than k results), and
the Rust API had no delete at all. There was no way to update a vector
or to reclaim the space of deleted ones.

Internal ids are record positions in the vector files (ADR 0003), and
the HNSW and IVF indexes refer to them directly.

## Decision

### Delete: tombstones, ids stay stable

- `SrvDB::delete(id)` records a tombstone and removes the metadata entry.
  The vector stays on disk; no other id changes.
- Tombstones live in a `redb` table next to the metadata, written with
  `Durability::None` and made durable by `persist()` like any other write.
  Recovery (ADR 0003) also drops tombstones for ids that were never
  committed.
- Every search path skips tombstoned ids while it searches, not by
  filtering a finished top-k list: flat, SQ8 and PQ scans skip them;
  HNSW still traverses deleted nodes (they keep the graph connected) but
  never returns them. A search returns `k` results whenever at least `k`
  live vectors exist.
- `count()` keeps meaning the number of stored records (it assigns the
  next id); `live_count()` is the number of searchable vectors.

### Update

`SrvDB::replace(id, vector, metadata)` tombstones `id` and appends the
new vector, returning its new id. Python `upsert(ids, ...)` keeps each
string id pointing at the newest vector, so string ids never change.

### Compaction: explicit and crash-safe

`compact()` rewrites storage without tombstoned records and renumbers ids
densely. It runs only when called. Because it changes the vector file and
`redb` together, it uses a two-step protocol:

1. Write the compacted vector file to `<name>.compact` and fsync it.
2. In one durable `redb` transaction: renumber metadata, clear tombstones,
   set the new committed count and set `compact_pending`.
3. Rename `<name>.compact` over the vector file, fsync the directory,
   delete the index files (derived data) and clear `compact_pending`.

On open, `compact_pending` means step 2 committed: finish step 3 (rename
the file if it is still there, delete index files, clear the flag).
Without the flag, a leftover `<name>.compact` is an abandoned attempt and
is deleted. A crash at any point therefore leaves either the old or the
new state.

Python string ids are stored in metadata (`__id__`), so the bindings
rebuild their id maps from metadata after compaction; Rust callers that
keep internal ids must re-read them after `compact()`.

## Alternatives considered

- **Physical delete on every call:** shifts every later id and forces an
  index rebuild per delete.
- **Filtering results after search:** returns fewer than `k` results and
  degrades badly as deletes accumulate.
- **Automatic background compaction:** surprises callers whose internal
  ids change; SrvDB has no background threads.

## Consequences

- Deleted vectors keep using disk and memory until `compact()`.
- HNSW and IVF indexes are rebuilt after compaction, which takes time
  proportional to the data.
- A crash test covers deletes and compaction like other writes.
