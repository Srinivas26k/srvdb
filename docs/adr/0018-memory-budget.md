# ADR 0018: Memory budget for automatic quantization

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Issue #86 asks for `srvdb.create(path, dim, index="auto", memory_mb=50)`:
the database picks how many bits it keeps per dimension so that its
resident footprint stays under a budget, changes that choice when growth
would break the budget, and reports what it chose.

Constraints:

- Indexes are derived data (AGENTS.md section 4). `vectors.bin` is the
  source of truth; RaBitQ codes are rebuilt from it (ADR 0008).
- Any on-disk change bumps a format version, keeps the old reader and
  ships a migration test (AGENTS.md section 5). The mode is recorded in
  `index_settings.bin` (ADR 0011).
- No numbers are published from an unidle machine (AGENTS.md section 11):
  this ADR states layouts and rules, not measured recall.

"Resident" here means the memory a search needs to keep in RAM to run at
full speed: heap allocations, plus mapped file pages every search reads.
Mapped pages that searches do not read can be evicted by the kernel and
are not counted.

## Decision

### Resident bytes per mode

`D` is the dimension, `n` the number of stored records (tombstoned
records occupy memory until `compact()`, so they count). The estimates
live in `tuning/budget.rs` and are the ones `memory_report()` returns.

| Mode | Per vector | Fixed | Source of the figure |
|------|-----------|-------|----------------------|
| flat | `4D` | 0 | `vectors.bin` records; a flat scan reads every one |
| sq8 | `D` | `8D` (min and max per dimension) | `sq8_vectors.bin` records, `ScalarQuantizer` |
| pq | `m` (`D/8`, else `D/4`) | `1024 D` (256 centroids of `D/m` f32 per sub-space) | `quantized_vectors.bin` records, codebooks |
| rabitq `b` | `ceil(D b / 8) + 8` | `4 D^2` (rotation) | fast-scan copy (ADR 0012) + factor + `|g|` width (ADR 0017) |
| hnsw graph | `16 m + 96` on top of the vectors | 0 | level-0 list of `2m` u64, layer `Vec`s, hash-map slot |
| ivf | hnsw graph term | `4 D nlist` (centroids) | partition graphs, centroids |
| links | `64` per link | 0 | each link sits in two `BTreeSet<(u16, u64)>` |
| keyword | `64` per (document, term) pair + `64` per document | 0 | postings, lengths, forward map |

RaBitQ details:

- The fast-scan copy is laid out in blocks of 32 vectors (ADR 0012), so
  the code term is `ceil(n / 32) * 32 * ceil(D b / 8)`; the factor and
  the unit width `w` (4 bytes each, ADR 0017) are per vector.
- The full vectors stay in `vectors.bin` but are read only for the `k * r`
  rerank candidates of each query, so they are not resident in this
  model. The mapped `rabitq_codes.bin` pages are touched while the copy is
  built (ADR 0012) and then evictable; not counted either.
- The rotation is `D x D` f32, regenerated on every open (ADR 0008):
  9 MiB at `D = 1536`. It dominates small databases of large dimension,
  which is why it is part of the selection.

Not counted anywhere: redb's page cache, transient per-query buffers,
and the Python binding's string-id maps (they live in `srvdb-py`).

The estimates are checked against measured allocations where they can
be measured: a counting global allocator in
`crates/srvdb/tests/test_memory_estimate.rs` measures the live heap of
the RaBitQ copy and rotation and of the HNSW graph across database sizes.
RaBitQ is measured against a flat database of the same vectors, so
redb's cache cancels. Stated tolerance: RaBitQ codes and rotation within
2 % (they match to the byte in practice), HNSW within 35 % (neighbour
lists grow by `push`, so their capacity depends on insertion order; the
estimate errs high). Flat,
SQ8 and PQ are record sizes of mapped files and exact by construction;
links and keyword are rough and are not used to pick a bit width, only
added to the total.

### Growth slack

`Vec` doubles its capacity on growth, so after many appends the
fast-scan copy could hold up to twice its length. Its buffers now grow
by at least 25 % with `try_reserve_exact`, which keeps appends amortised
O(1) and bounds the slack to 25 % of the copy. On reopen the copy is
allocated exactly.

### Selection rule

- Budget: `memory_mb` MiB, `1..=1048576`. Limit `L = memory_mb * 2^20 *
  0.8`; the 20 % headroom covers the growth slack above (`1 / 1.25 =
  0.8`) and what the model does not count.
- Ladder, highest fidelity first: flat (exact), RaBitQ 4, 2, 1 bit. All
  of them keep `vectors.bin`, so moving down the ladder is a rebuild of
  derived data, never a loss of vectors.
- SQ8 and PQ are not on the ladder. They keep no full-precision vectors,
  so a database that grows past them cannot be re-encoded to fewer bits,
  and they search without an exact rerank. RaBitQ at 4 bits
  (`D/2 + 8` bytes) is smaller than SQ8 (`D` bytes) and reranks exactly.
  They stay available as fixed modes, and `memory_report()` reports them.
- Choice: the first ladder mode `t` with
  `fixed(t) + per_vector(t, n') + links + keyword <= L`, where `n'` is
  the record count after the write being made (`0` at create).
- Projection: `memory_report()` returns `capacity`, the record count the
  current mode holds within `L`, so callers see when the next step down
  will happen. The ladder only goes down: deleting does not free memory
  until `compact()`, and stepping up is left for later.

### When growth would exceed the budget

Every `add` / `add_batch` checks the rule for `n'` before anything is
written.

- If the current mode still fits, nothing changes.
- If a lower mode fits, the database re-encodes before appending:
  1. `attach_rabitq` with the new bit width: `RabitqStorage::create`
     rewrites `rabitq_params.bin` atomically and re-encodes every stored
     vector from `vectors.bin` into a new `rabitq_codes.bin`, then fsyncs.
  2. `index_settings.bin` is rewritten atomically with the new mode.
  3. The vectors are appended as usual.
  A crash after 1 leaves settings naming the old mode: an old RaBitQ
  width sees params of another width and rebuilds (ADR 0008); old flat
  ignores the codes. The next write steps down again. A crash after 2
  leaves the new mode with complete codes. Settings are always written
  before any vector that needed them can be committed, so the committed
  count always fits the recorded mode.
  The step down is kept even if the append after it fails (I/O,
  metadata): the database then holds fewer bits than its count needs,
  which costs precision, never data or the budget; this is documented on
  `add`. If the re-encode itself fails, the previous width (or flat) is
  restored and nothing is recorded or appended.
  `crates/srvdb/tests/test_crash_budget.rs` kills a writer stepping down
  at random moments and checks that invariant on every reopen.
- If even 1 bit does not fit, the write is refused with the new
  `Error::MemoryBudget` (Python `ValueError`), nothing is written, and the
  message names the bytes needed and the limit, and recommends a larger
  `memory_mb`, `compact()` after deletes, or a fixed mode.

### Persistence in `index_settings.bin`

The settings payload gains `memory_mb: u64` (0 = no budget) and its
format goes from 1 to 2. The recorded kind is the mode in use (flat or
RaBitQ with its bit width), so `open_existing` reopens it as is and then
applies the budget.

- Databases without a budget are still written as format 1, so earlier
  0.4 builds keep reading them. Format 2 is written only with a budget.
- The reader accepts both: format 1 reads as "no budget". Format 3 and up
  are `Corrupt` ("upgrade SrvDB"), as before.
- A budget with a kind other than flat or RaBitQ, or outside the range,
  is `Corrupt`.
- Migration test: a format-1 image written by the previous encoder opens
  unchanged and is rewritten as format 1.

### API

- Rust: `CreateOptions::new(IndexChoice::Auto).set("memory_mb", 50)`;
  `memory_mb` is accepted only for `auto` and range-checked in `create`.
  `SrvDB::memory_report() -> MemoryReport` (non-exhaustive struct: mode,
  bits, count, dimension, vector / index / links / keyword / total bytes,
  budget, limit, capacity, exact).
- Python: `srvdb.create(path, dim, index="auto", memory_mb=50)` (options
  are passed to the core unchanged) and `db.memory_report()` returning a
  dict with the same fields.
- `auto` without `memory_mb` keeps its existing behaviour.

## Alternatives considered

- **Re-encode to SQ8 or PQ.** Discards the full vectors; a later step
  down or a rerank would be impossible. Rejected for the ladder.
- **Refuse only, never re-encode.** Simpler, but the step down reuses the
  rebuild path ADR 0008 already relies on and tests, so it costs little.
- **Count the mapped `vectors.bin` in RaBitQ mode.** RaBitQ would then
  never be smaller than flat, and searches do not read those pages except
  for rerank candidates.
- **Read RSS from the OS.** Platform-specific, includes the host
  application, and cannot predict the footprint at a future count.
- **Persist the budget in its own file.** One more file to validate and
  back up for one integer; the settings record already has a format
  number for this.

## Consequences

- A budgeted database can change mode on `add`; the write that crosses a
  step pays a full re-encode, at most three times over its life.
- `Error` gains `MemoryBudget` (the enum is `#[non_exhaustive]`).
- Expected recall per bit width is not stated here: it needs a benchmark
  run on an idle machine. The committed recall test checks ordering only.
- The estimates must be kept in step with the layouts they describe; the
  allocation test fails when a layout grows.
