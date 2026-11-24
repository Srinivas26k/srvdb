# ADR 0017: Certified and deadline search for RaBitQ

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

RaBitQ search (ADR 0008, ADR 0012) scans the codes for `k * r`
candidates and rescores them on the full vectors. The rerank factor `r`
is a fixed guess. It says nothing about how far the result is from exact
search, and the scan always runs to the end. Issue #81 asks for two
things. First, use the RaBitQ error bound to rescore only the candidates
whose rank is in doubt, and report a confidence. Second, accept a time
budget and report how complete the result is.

## Decision

### Error bound

Let `o` be a unit data vector, `y = R o` its rotated copy, `g` its grid
vector and `o_bar = g / |g|`. The stored factor is `f = 1 / <g, y>`. For
a unit query `q`, the estimate is `est = <o_bar, q'> / <o_bar, o>` (with
`q' = R q`). Write `q = s o + sqrt(1 - s^2) e`, where `s = <o, q>` and
`e` is a unit vector orthogonal to `o`. Then

    est - s = sqrt(1 - s^2) * <o_bar, e> / <o_bar, o>
            = sqrt(1 - s^2) * sqrt(1 / <o_bar, o>^2 - 1) * Z

`Z` is the component of `R e` along the part of `o_bar` orthogonal to
`y`. The RaBitQ analysis (Gao and Long, arXiv:2405.12497, Theorem 3.2
and its proof) applies to any code that is a deterministic function of
`y`. Given `y`, the rotation leaves `R e` uniform on the unit sphere of
the `D - 1` dimensional complement of `y`. So `Z` is one coordinate of
a uniform unit vector in `D - 1` dimensions, and by spherical
concentration `P(|Z| > eps / sqrt(D - 1)) <= 2 exp(-eps^2 / 2)`. This
holds for 1, 2 and 4 bits because the argument does not depend on the
grid.

`<o_bar, o> = <g, y> / |g| = 1 / (f |g|)`. The file stores only `f`, so
`|g|^2` is summed from the codes once, when the in-memory fast-scan copy
is built or appended (ADR 0012). Each vector gets a unit width
`w = sqrt(f^2 |g|^2 - 1) / sqrt(D - 1)`. That is 4 more bytes per
vector of resident memory, still derived data, with no format change.

The scan does not compute `est` exactly. It uses the u8 tables, which
are off by at most `|f| * M * step / 2` (ADR 0012, `QueryLut::max_error`).
Call that `lut`, and add a fixed `1e-3` for f32 rounding in the
rotation, the tables and the factor. The interval half-width is

    e0 = eps * w
    h  = e0 * sqrt(1 - s_min^2) + lut + 1e-3,
    s_min = clamp(|est| - e0 - lut, 0, 1)

`s_min` is the smallest `|s|` the same interval allows. On the event
`|Z| <= eps / sqrt(D - 1)` we have `sqrt(1 - s^2) <= sqrt(1 - s_min^2)`,
so this tighter width is valid on the same event. It matters for
near-duplicates, which are the vectors that decide the top k. A unit
test checks that every estimate lies inside its interval for 1, 2 and 4
bits.

### Choice of eps and meaning of confidence

The union bound over all `n` live vectors fails with probability at most
`n * 2 exp(-eps^2 / 2)`. SrvDB sets this to `1 - c` with
`c = CERTIFIED_CONFIDENCE = 0.99`:

    eps = sqrt(2 ln(2 n / (1 - c)))

which is about 5.8 at `n = 100 000`.

**Confidence** is a lower bound on the probability, over the random
rotation seed, that the certified result is exactly the top k of the
scanned vectors (ties aside). It follows that expected recall@k against
exact search over those vectors is at least the confidence. A query
reports `confidence = 0.99`. With a deadline, confidence still describes
the scanned vectors, and `completeness` says how many vectors that was.
The two values are kept separate because nothing is known about the
vectors that were not scanned.

The bound is conservative. It does not depend on the data distribution,
it applies a union bound over every vector, and the concentration
inequality is not tight. The test measures recall against exact search
on 24 seeded datasets (8 seeds, 3 bit widths, cluster count and spread
varying with the seed) and requires mean recall >= the reported
confidence on every run.

### Certification

Each scanned vector gets an interval `[lo, hi] = est -/+ h`.

1. Let `tau` be the k-th largest `lo`. At least `k` vectors score
   `>= tau`, so any vector with `hi < tau` is outside the top k and is
   dropped. Each parallel scan task applies this rule with its own
   `tau`, which is never above the global one, so the union of the
   tasks' survivors is a superset of the global survivors (`C`).
2. A candidate `x` in `C` is **certified** when `lo_x >= tau` and at
   most `k - 1` other candidates have `hi > lo_x`. Any vector that
   truly beats `x` has `hi > lo_x >= tau`, so it is in `C` and is
   counted. Therefore `x` is in the top k. At most `k` candidates are
   certified; exact ties beyond `k` stay uncertain.
3. Only the **uncertain** candidates (`C` minus the certified ones) are
   rescored on the full vectors. The result is the certified set plus
   the best `k - certified` uncertain candidates by exact score.

If every interval holds, this is exactly the top k. The certified
results are also scored exactly, which costs at most `k` vector reads,
so that scores and ordering match every other mode. Certification
settles membership, not the returned score. The rerank factor `r` is
not used: the number of vectors rescored follows from the intervals.
With 1 bit the intervals are wide, and most candidates near the top are
uncertain. Certification saves the most at 4 bits.

### Deadline

`deadline` is a `Duration` measured from the start of the call. Every
rayon scan task reads the clock before each 32-vector block and stops
once the deadline has passed. This is cooperative: no thread is
interrupted. The rerank after the scan always runs to completion, so a
call can overrun the deadline by the time that rerank takes. Results are
the best found among the scanned vectors. `completeness` is the number of
scanned live vectors divided by the number of live vectors (1.0 when
there are none). A deadline of 0 scans nothing and returns no results
with completeness 0. The deadline works with or without certification.
With a deadline that never passes, an uncertified search runs the same
code path as `search` and returns identical results; a test checks this.

### API

- Rust: `SrvDB::search_rabitq_with(&Vector, k, &RabitqSearch) ->
  Result<SearchOutcome>`, where `SearchOutcome { results, stats:
  SearchStats { completeness, confidence, certified, reranked, scanned } }`.
  `RabitqSearch` has the fields `certify` and `deadline`. All three
  types are `#[non_exhaustive]`. Any non-RaBitQ database returns
  `InvalidInput`.
- Python: `db.search(q, k, certify=True)` and/or `deadline_ms=...`
  returns a `SearchOutcome`. It iterates, indexes and has a length like
  the usual list of `(id, score)` tuples, and also has `completeness`,
  `confidence`, `certified`, `reranked` and `scanned` attributes. On
  any other mode, or combined with `where`, it raises `ValueError`.
  Without these arguments `search` is unchanged.

## Alternatives considered

- **Use the paper's empirical `eps0 = 1.9`.** It gives tighter
  intervals, but it is not a guarantee at a stated confidence. We chose
  the bound we can state.
- **Report confidence per query from the data.** Any data-dependent
  estimate would need a calibration set; the seed-probability bound needs
  none.
- **Store `|g|` in the code file.** That is a format change (version
  bump and migration) for a value that is cheap to derive on open.
- **Interrupt the rerank too.** A partial rerank leaves results without
  exact scores. The scan is the O(n) part; the rerank is bounded by the
  candidate set.

## Consequences

- Resident memory for RaBitQ grows by 4 bytes per vector (the widths).
- Certified search can rescore more vectors than `k * r` at 1 bit and
  fewer at 4 bits. We publish no latency numbers until they are measured
  on an idle machine with a committed benchmark (AGENTS.md section 11).
- Changing `CERTIFIED_CONFIDENCE` or the bound changes the reported
  confidence, which is a documented contract. Such a change needs an
  update to this ADR.
