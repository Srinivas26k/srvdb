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
- Limits, checked before anything changes (`InvalidInput` otherwise)
  and again while loading (`Corrupt` as soon as one is exceeded):
  - at most **1,024 outgoing links per record** (fan-out), all types
    together;
  - at most **100,000 incoming links per record**, so deleting or
    replacing a hub stays bounded while a popular record can still be
    cited widely;
  - at most **1,000,000 links per database**;
  - at most **1,024 distinct link type names per database**.

### Memory

The whole graph is held in memory: an outgoing and an incoming adjacency
map of `BTreeSet<(u16, u64)>`, with link type names interned once (a
`u16` id per type plus a name table). Measured by
`tests/test_links_memory.rs` (100,000 links over 10,000 records, counting
allocator): **about 97 bytes per link**, an upper bound because it also
counts what redb keeps allocated after the commit. At the 1,000,000-link
cap that is about 100 MB. A higher cap would need an on-disk adjacency
and its own ADR. Deleting a record takes its own adjacency sets out of
the maps; it never copies the graph.

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

Every commit and compaction transaction also records a **stamp** of the
state it commits, `links_stamp` in the `state` table: a 64-bit hash of
the committed record count and the tombstone set (count of tombstones
plus an order-independent sum of mixed ids). It is written even when
there are no link changes. A SrvDB without links (0.3 and earlier) never
updates it, so any add, delete or compaction made by such a version is
visible as a stamp that no longer matches the committed state.

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

Nothing is dropped or repaired silently. Each of these is `Corrupt`:

- a `links_format` other than 1 (written by a newer SrvDB);
- a non-empty `links` table whose stamp is missing or does not match the
  committed count and tombstones. The message says the links were written
  before an older SrvDB modified the database and may point at the wrong
  records, and names the recovery call;
- a link with an id at or beyond the committed count, a self-link, an
  end that is tombstoned, or an invalid type name;
- more links than a limit allows, detected while iterating.

Recovery is explicit: `SrvDB::drop_links(path)` (Python
`srvdb.drop_links(path)`) takes the database lock, deletes every link
and writes a fresh stamp in one durable transaction. Vectors and
metadata are not touched. It never runs on its own.

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
needed for them. A database from before links (SrvDB 0.3 and earlier)
has no `links` table; it opens with no links and gets the table and the
stamp with its first commit. The migration test opens the committed
SrvDB 0.3 fixture.

Downgrade: SrvDB 0.3 opens a database with links but ignores them. If it
adds, deletes or compacts records, the stamp no longer matches and this
version refuses to open the database until `drop_links` is called (the
compatibility tests simulate 0.3 by calling `MetadataStore::compact` and
`MetadataStore::commit` directly). Opening it read-only with 0.3 and
writing nothing keeps the links valid.

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
- The whole graph is in memory; the 1M-link limit bounds it at about
  100 MB (measured upper bound, see Memory).
- A database touched by SrvDB 0.3 after links were saved needs an
  explicit `drop_links` before it opens; links are never silently lost.
- `persist()` writes pending link operations; a bulk of link calls costs
  one transaction, not one per link.
- Every engine path that changes ids (delete, replace, compact) must keep
  the graph in step; the delete-cascade, replace and compaction tests
  guard this.
