# ADR 0016: Typed links between records and k-hop search expansion

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Applications built on SrvDB often know how their records relate: a paper
cites another, a chunk belongs to a section, a ticket duplicates a ticket.
Issue #80 asks for typed, directed links between records
(`db.link(a, "cites", b)`) and for search that follows them
(`db.search(q, k, expand=("cites", 2))`).

Constraints:

- Links are written by the user and cannot be recomputed from the vectors.
  They are **primary data**, unlike HNSW and IVF (ADR 0003), and need the
  same guarantee as records.
- Records are addressed by internal ids that compaction renumbers
  (ADR 0004); links refer to those ids.
- Search must stay bounded: a dense link graph must not turn one query
  into a scan of the whole database.

## Decision

### Data model

- A link is a triple `(source id, link type, target id)`. It is directed:
  `a -cites-> b` says nothing about `b -cites-> a`.
- A link type is a name matching `[a-z][a-z0-9_]{0,63}` (lowercase ASCII
  letter, then up to 63 lowercase letters, digits or underscores). It is
  checked against this allow-pattern, never interpreted.
- Both ends must be existing, not deleted records. Self-links are
  rejected. Linking twice is a no-op; `link` returns whether the link is
  new and `unlink` whether it existed.
- Limits, checked before anything changes (`InvalidInput` otherwise):
  - at most **1,024 outgoing links per record** (fan-out), all types
    together;
  - at most **10,000,000 links per database** (the graph is held in
    memory, about 100 bytes per link);
  - incoming links are not limited per record: a popular record may be
    cited by many others.

### Storage

Links live in `metadata.db`, the redb database that already holds the
commit point, in a new table `links` with key `(u64 source, &str type,
u64 target)` and an empty value. redb checksums its pages, so links get
the same integrity checking as metadata. A `state` entry `links_format = 1`
records the table's format.

Links are **not** written when `link` is called. Like deletes (ADR 0004),
they are kept as a list of pending operations in memory and written by the
next `persist()` **inside the same durable transaction** that records the
committed vector count and tombstones. One transaction means links and
records are atomic: a link can never refer to a record that was not
committed with it.

The full graph is loaded into memory on open (an outgoing and an incoming
adjacency map), so `neighbors` and expansion never touch disk.

### What is durable when

> When `persist()` returns, every earlier `link` and `unlink` survives a
> crash or power loss, together with every earlier add and delete. After a
> crash, the database reopens with exactly the links of the last successful
> `persist()`.

- Links take effect in memory as soon as `link` returns: `neighbors` and
  expansion see them before `persist()`.
- Links made after the last `persist()` are lost on reopen, whether the
  process crashed or exited cleanly (they were never written).
- If the commit transaction fails, the pending operations stay queued and
  are written by the next successful `persist()`.

### Delete, replace and compaction

- **Delete** removes every link the record is part of, outgoing and
  incoming, in memory at once and on disk with the delete in the next
  `persist()`. A deleted record never appears in `neighbors` or in an
  expansion.
- **Replace** (`SrvDB::replace`, Python `upsert`) moves the old record's
  links, in both directions, to the new record: the record keeps its
  relationships. `SrvDB::move_links` exposes the same operation.
- **Compaction** renumbers ids. Links are rewritten to the new ids in the
  same durable redb transaction that renumbers metadata (ADR 0004 step 2),
  so an interrupted compaction finishes or is discarded with its links.
  Links to deleted records cannot exist at this point (deletes removed
  them).

### Validation on open

Every stored link is checked: both ids below the committed count, a valid
type name, no self-link. A violation is `Corrupt` naming the problem; a
`links_format` other than 1 is `Corrupt` saying the database was written
by a newer SrvDB. A link that touches a tombstoned record (possible only
if an older SrvDB, which ignores the table, deleted the record) is dropped
with a warning and the drop is written by the next `persist()`.

### Search expansion

`search_expand(query, k, expansion)` runs the normal search for the top
`k` records (hop 0), then follows outgoing links of one type breadth-first
for up to `hops` steps:

- `hops` is 1 to 3; the number of expanded records added is capped by
  `max_results` (default 100, at most 1,000).
- Each record appears once, at the smallest hop that reaches it.
- An expanded record carries the score of the record it was reached from
  (the best-scoring one at the previous hop); results are ordered by hop,
  then score. Scores of expanded records are therefore inherited, not
  their own similarity to the query. Fused ranking is left for a later
  ADR with a benchmark.

### Format and compatibility

The existing files and tables are unchanged, so no version bump is
needed for them. A database without a `links` table (SrvDB 0.4 and
earlier) gets an empty one on first open; the migration test opens the
committed SrvDB 0.3 fixture. Downgrade caveat: an older SrvDB opens a
database with links but ignores them; deleting or compacting there leaves
stale links, which this version drops (tombstoned ids) or rejects as
`Corrupt` (ids beyond the count after compaction). Do not compact with an
older version after using links.

## Alternatives considered

- **Write links with `Durability::None` at `link` time** (like metadata):
  redb persists those on a clean shutdown, so whether an unpersisted link
  survives would depend on how the process exited. Rejected for the same
  reason as for tombstones in ADR 0004.
- **A separate links file**, replaced atomically on persist: needs its
  own ordering against the redb commit point, and a crash between the two
  could leave links to uncommitted records. One redb transaction avoids
  that protocol entirely.
- **Store links in the metadata JSON:** no reverse index for delete
  cascade, and every link change rewrites user metadata.
- **Unbounded hops or results:** a single query could traverse the whole
  graph. Bounds are cheap to relax later and hard to add.

## Consequences

- Links are as safe as records and need no new recovery code beyond
  validation on open.
- The whole graph is in memory; the 10M-link limit bounds that.
- `persist()` writes pending link operations; a bulk of link calls costs
  one transaction, not one per link.
- Every engine path that changes ids (delete, replace, compact) must keep
  the graph in step; the delete-cascade, replace and compaction tests
  guard this.
