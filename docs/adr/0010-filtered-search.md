# ADR 0010: Filtered search during HNSW traversal

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Before 0.4 SrvDB had no filter API. Callers ran `search(k)` and dropped
non-matching results, so a selective filter returned fewer than k results.
In the benchmark harness (`--filter-pct`, 50k clustered 128-d vectors,
k = 10) post-filtering left every query short at 1% and 10% matching, with
recall@10 of 0.008 and 0.09 against exact filtered search (issue #25).
Hardware, OS, commit and command for this and every number below are in
docs/benchmarks.md, "Filtered search".

Constraints: indexes are derived data (ADR 0004), so nothing new may be
persisted; metadata lives in redb as JSON text, and reading it per visited
node would dominate the cost of a search.

## Decision

1. **The filter is a set of ids.** `SrvDB::search_filtered(query, k,
   &allowed)` takes a `HashSet<u64>`; `SrvDB::ids_where(predicate)` builds
   it from metadata in one pass. The caller reuses the set across queries
   while the filter and the data are unchanged. Membership is an O(1) hash
   lookup during traversal, and the number of matches, and so the
   selectivity, is known before the search starts.
2. **ACORN-1 traversal.** `HNSWIndex::search_accepted` walks the graph with
   an accept predicate. Rejected nodes (deleted or not in the set) still
   navigate but never enter the result heap, and the search runs until it
   holds `ef` accepted results. When fewer than M of an expanded node's
   layer-0 links are accepted, it also scores accepted neighbours of its
   rejected neighbours (two-hop expansion, Patel et al. 2024), capped at M
   per node so the expansion costs O(M^2) per node. Without the cap the
   two-hop scan cost seconds per query on 20k vectors.
3. **Exact scan when it is cheaper.** HNSW skips the graph and scores the
   matching ids exactly when `matches^2 < 4 * ef * M * live`. Equal cost
   at selectivity `s` is `s * n * c_exact = c_walk(s)`. The ignored test
   `filtered_walk_cost` (`cargo test --release -p srvdb filtered_walk_cost
   -- --ignored --nocapture`) forces the walk on 20k seeded clustered 64-d
   vectors (M 16, ef 100). On an AMD Ryzen 5 3550H, Ubuntu 26.04.1,
   kernel 7.0.0-43, commit 08801ab, it measured 760, 2.4k, 7.8k and 19.1k
   visits per query at 100%, 50%, 10% and 1% matching. Each visit cost
   0.4 to 4 us, rising as the filter gets more selective (two-hop
   expansion), against 0.14 to 0.21 us per exactly scored id. That gives
   a crossover ratio of 1.1 to 4.1, so the ratio is 4, the top of the
   selective range. With the defaults (ef 50, M 16) the exact scan is used
   below about 18% matching at 100k vectors, 5.7% at 1M and 1.8% at 10M.
   A fixed percentage would be wrong for either small or large databases.
   An earlier probe on uniform random data suggested a ratio of 15; it
   was not committed and was replaced by this measurement.
4. **k results guaranteed.** If the walk returns fewer than
   `min(k, matches)` results (an unreachable part of the graph), the
   matches are scored exactly. Flat, SQ8, PQ and IVF modes always score
   the matches exactly, using the same storage as unfiltered search.

## Alternatives considered

- **A metadata predicate evaluated inside the search.** It would read and
  parse JSON from redb for every visited node, often thousands per query,
  and selectivity would be unknown up front, so the exact-scan choice
  could not be made. A predicate is still offered, but through
  `ids_where`, which runs once.
- **Post-filtering with oversampling.** Still returns fewer than k for
  selective filters, and the oversampling factor depends on the data.
- **A fixed selectivity cutoff (for example 1%).** The crossover depends
  on n, ef and M; the measurements above show a fixed cutoff is too low
  for small databases and too high for large ones.
- **Full ACORN (denser graph built with M * gamma links).** Changes the
  index build and its memory use, and needs a format change; not
  justified by current numbers.

## Consequences

- Filtered search returns k results whenever k live vectors match, with
  exact results at low selectivity.
- Two new public methods on `SrvDB` are a compatibility promise.
- Nothing is persisted; the index format is unchanged.
- Open points:
  - **Python API.** Python has no filter API yet; the framework adapters
    (ADR 0009) still post-filter and can move to `search_filtered` once
    one exists.
  - **Cost per visit.** A visited node costs far more than its distance
    call (HashMap node lookup, SipHash visited set). Reducing that moves
    the threshold and needs its own benchmark.
  - **IVF.** IVF scores the matches exactly instead of walking partitions
    with the filter.
  - The ratio of 4 comes from one machine under load and one dataset;
    re-measure on a quiet machine, on real datasets and at 1M vectors.
