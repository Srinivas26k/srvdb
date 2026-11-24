---
title: What is RaBitQ quantization?
description: How SrvDB's RaBitQ mode works, step by step - random rotation, 1/2/4-bit codes, the nibble lookup-table fast-scan and exact rerank - with sizes per vector and when to choose it.
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: '{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[{"@type":"Question","name":"Does RaBitQ in SrvDB need training?","acceptedAnswer":{"@type":"Answer","text":"No. The rotation comes from a fixed 64-bit seed and each vector is encoded on its own, so the first vector can be added straight away."}},{"@type":"Question","name":"Are RaBitQ search scores exact in SrvDB?","acceptedAnswer":{"@type":"Answer","text":"With rerank on, which is the default, the returned scores are exact cosine similarities computed from the full vectors. With rerank set to 0 they are estimates from the codes."}},{"@type":"Question","name":"Does RaBitQ make a SrvDB database smaller?","acceptedAnswer":{"@type":"Answer","text":"No. Full vectors are kept for rerank, so a RaBitQ database is slightly larger than a flat one. Use SQ8 or PQ when disk space is the limit."}},{"@type":"Question","name":"What happens if the RaBitQ code files are damaged?","acceptedAnswer":{"@type":"Answer","text":"The codes are derived data. If they are missing, fail validation or were made with other settings, SrvDB rebuilds them from vectors.bin when the database opens."}}]}'
---

RaBitQ is a training-free way to compress each vector to 1, 2 or 4 bits
per dimension after a random rotation, with one correction factor that
turns a dot product with the codes into a cosine estimate. In SrvDB it is
a scan accelerator: the codes pick `k × rerank` candidates fast, and the
full vectors, which are always kept, give the exact final scores.

The method follows RaBitQ (Gao and Long, SIGMOD 2024) and its multi-bit
extension (SIGMOD 2025). The design is recorded in
[ADR 0008](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0008-rabitq-quantization.md)
(quantizer and files) and
[ADR 0012](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0012-rabitq-fast-scan.md)
(fast-scan). For the how-to, see the [RaBitQ guide](/srvdb/guides/rabitq/).

## How it works

<figure class="concept-diagram">
<svg viewBox="0 0 720 200" role="img" aria-labelledby="rbq-pipe-t">
<title id="rbq-pipe-t">RaBitQ pipeline: rotate, encode to bit codes, scan with lookup tables, rerank on full vectors</title>
<defs><marker id="rbq-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<rect class="dg-box" x="10" y="30" width="130" height="70" rx="6"/>
<text x="75" y="58" text-anchor="middle">1. rotate</text>
<text x="75" y="80" text-anchor="middle" class="dg-muted">y = R x / |x|</text>
<rect class="dg-box" x="190" y="30" width="150" height="70" rx="6"/>
<text x="265" y="58" text-anchor="middle">2. encode</text>
<text x="265" y="80" text-anchor="middle" class="dg-muted">b-bit codes + f</text>
<rect class="dg-hi" x="390" y="30" width="150" height="70" rx="6"/>
<text x="465" y="58" text-anchor="middle">3. fast-scan</text>
<text x="465" y="80" text-anchor="middle" class="dg-muted">nibble tables</text>
<rect class="dg-box" x="590" y="30" width="120" height="70" rx="6"/>
<text x="650" y="58" text-anchor="middle">4. rerank</text>
<text x="650" y="80" text-anchor="middle" class="dg-muted">exact cosine</text>
<path class="dg-arrow" d="M140,65 H186" marker-end="url(#rbq-a)"/>
<path class="dg-arrow" d="M340,65 H386" marker-end="url(#rbq-a)"/>
<path class="dg-arrow" d="M540,65 H586" marker-end="url(#rbq-a)"/>
<text x="563" y="56" text-anchor="middle" class="dg-muted">k×r</text>
<rect class="dg-box" x="10" y="140" width="330" height="40" rx="6"/>
<text x="175" y="165" text-anchor="middle">vectors.bin (full f32, source of truth)</text>
<rect class="dg-box" x="390" y="140" width="150" height="40" rx="6"/>
<text x="465" y="165" text-anchor="middle">rabitq_codes.bin</text>
<path class="dg-dash" d="M75,140 V100"/>
<path class="dg-dash" d="M465,140 V100"/>
<path class="dg-dash" d="M300,180 V192 H650 V100"/>
</svg>
<figcaption>Encoding (steps 1 and 2) happens on add. A search runs steps 3 and 4: the scan reads only codes, rerank reads k × rerank full vectors.</figcaption>
</figure>

### 1. Rotate

Each vector `x` is scaled to unit length and multiplied by a random
orthogonal matrix `R`: `y = R x / |x|`. After rotation all coordinates
look alike, so a few bits each lose little. `R` is generated from a 64-bit
seed with SrvDB's own SplitMix64 and Box-Muller code, orthonormalised in
f64. A CRC32C fingerprint of the matrix is stored in `rabitq_params.bin`,
so a seed that ever expands to a different matrix is detected and the
codes are rebuilt. `R` is dense (`dimension × dimension`), which is why
RaBitQ mode refuses dimensions above 4096.

### 2. Encode

Code `c_i` is in `[0, 2^b)` and stands for the grid value
`g_i = c_i + 1/2 - 2^(b-1)`. SrvDB tries 32 scales `t` per vector and
keeps the one that best aligns `g` with `t × y` (maximises
`<g, y> / |g|`). With 1 bit the codes are just the signs of `y`. One f32
factor `f = 1 / <g, y>` is stored with the codes. No data is learned from
other vectors: there is no training step and no centroid.

A record in `rabitq_codes.bin` is `f` (4 bytes, little-endian) followed by
the codes packed `8 / b` dimensions per byte, lowest bits first.

### 3. Fast-scan with nibble lookup tables

The query is rotated the same way, `q' = R q / |q|`, and the cosine is
estimated as `f × <g, q'>`. Instead of decoding codes, SrvDB splits each
code byte into two 4-bit nibbles. A nibble holds `4 / b` dimensions, so it
has only 16 possible values, and the query gets one 16-entry table per
nibble with the partial dot product for each value.

<figure class="concept-diagram">
<svg viewBox="0 0 720 210" role="img" aria-labelledby="rbq-lut-t">
<title id="rbq-lut-t">A code byte splits into two nibbles; each nibble indexes a 16-entry u8 table; 32 vectors are looked up at once</title>
<defs><marker id="rbq-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<text x="10" y="22" class="dg-muted">code byte j of vector v</text>
<rect class="dg-hi" x="10" y="32" width="80" height="34" rx="4"/>
<rect class="dg-box" x="90" y="32" width="80" height="34" rx="4"/>
<text x="50" y="54" text-anchor="middle">high</text>
<text x="130" y="54" text-anchor="middle">low</text>
<path class="dg-arrow" d="M170,49 H226" marker-end="url(#rbq-b)"/>
<text x="240" y="22" class="dg-muted">table for this nibble: 16 entries, u8</text>
<g>
<rect class="dg-box" x="240" y="32" width="28" height="34"/><rect class="dg-box" x="268" y="32" width="28" height="34"/><rect class="dg-hi" x="296" y="32" width="28" height="34"/><rect class="dg-box" x="324" y="32" width="28" height="34"/><rect class="dg-box" x="352" y="32" width="28" height="34"/><rect class="dg-box" x="380" y="32" width="28" height="34"/><rect class="dg-box" x="408" y="32" width="28" height="34"/><rect class="dg-box" x="436" y="32" width="28" height="34"/>
<text x="474" y="54" class="dg-muted">... 16</text>
</g>
<path class="dg-arrow" d="M310,66 V102" marker-end="url(#rbq-b)"/>
<text x="322" y="90" class="dg-muted">add to the sum for vector v</text>
<text x="10" y="128" class="dg-muted">in-memory block: byte j of 32 vectors side by side</text>
<g>
<rect class="dg-box" x="10" y="138" width="20" height="30"/><rect class="dg-box" x="30" y="138" width="20" height="30"/><rect class="dg-box" x="50" y="138" width="20" height="30"/><rect class="dg-box" x="70" y="138" width="20" height="30"/><rect class="dg-box" x="90" y="138" width="20" height="30"/><rect class="dg-box" x="110" y="138" width="20" height="30"/><rect class="dg-box" x="130" y="138" width="20" height="30"/><rect class="dg-box" x="150" y="138" width="20" height="30"/>
<text x="180" y="158" class="dg-muted">... v = 0..31</text>
</g>
<rect class="dg-hi" x="300" y="138" width="410" height="30" rx="4"/>
<text x="505" y="158" text-anchor="middle">one shuffle = 32 lookups (AVX2, NEON, or scalar)</text>
<path class="dg-arrow" d="M272,153 H296" marker-end="url(#rbq-b)"/>
<text x="10" y="196" class="dg-muted">sum × step + Σ table minimums → dot product → × f → cosine estimate</text>
</svg>
<figcaption>Tables are rounded to u8 with one shared step, so a CPU byte-shuffle can look up 32 vectors per instruction. Integer sums are identical across the AVX2, NEON and scalar kernels.</figcaption>
</figure>

- **u8 tables.** Each f32 table is shifted by its minimum and divided by
  one step shared by all tables (`step = largest range / 255`), then
  rounded. The dot product is `step × sum + sum of minimums`. Rounding adds
  at most `M × step / 2` to the dot product for `M` tables (two per code
  byte), scaled by the record's factor. Rerank removes this error from the
  returned scores.
- **Interleaved copy.** On open SrvDB copies the codes into blocks of 32
  vectors, byte `j` of every vector side by side, and keeps it in step on
  add and delete. The file format is unchanged; the copy is derived data.
- **Kernels.** AVX2 on x86_64 and NEON on aarch64 are picked at run time;
  other CPUs use the portable scalar kernel. A property test checks that
  every kernel produces the same integer sums.
- **Parallel.** Rayon tasks of 1024 vectors each keep their own top
  `k × rerank` candidates.

### 4. Rerank on full vectors

The `k × rerank` best estimates are rescored exactly against
`vectors.bin` and the top `k` are returned. Defaults are 40, 20 and 10 for
1, 2 and 4 bits: fewer bits give rougher estimates, so more candidates are
checked. Rerank is a search-time setting (`set_rabitq_rerank`), not stored
on disk; `0` returns the estimates without reranking.

### 5. Certified and deadline search (optional)

The estimate error has a known bound. For a stored vector with code
direction `o_bar`, the estimate differs from the true cosine `s` by
`sqrt(1 - s²) × sqrt(1/<o_bar, o>² - 1) × Z`, where `Z` behaves like one
coordinate of a random unit vector in `D - 1` dimensions (RaBitQ paper,
Theorem 3.2). SrvDB adds the u8 table rounding from step 3 on top, which
gives every scanned vector an interval `[lo, hi]` around its estimate.

`db.search(q, k, certify=True)` uses these intervals:

1. Vectors whose `hi` is below the k-th largest `lo` cannot be in the
   top k and are dropped.
2. A vector whose `lo` is above the `hi` of all but at most `k - 1`
   others is **certified**: it is in the top k without rescoring.
3. Only the remaining **uncertain** candidates are rescored on the full
   vectors. The rerank factor is not used.

The interval width is chosen so that, if the rotation were drawn at
random, every interval would hold with probability at least 0.99.
**Confidence** (0.99) is that nominal probability for this query. It is
**not** a guarantee for your query, **not** a long-run rate for your
database (its seed is fixed, and all queries share it), and **not** a
bound for a batch of queries together. It is also approximate: SrvDB's
rotation is a pseudo-random f32 approximation of a random one, and the
allowance for f32 rounding is a heuristic. The test suite checks that
mean recall against exact search is at least the reported confidence
on 24 seeded datasets.

**On clustered data, almost nothing is certified.** In those tests no
result was certified without rerank, and at 1 bit certified search
rescored several times as many candidates as a normal search. The rerank
is capped at `max(k × rerank, 1000)` candidates. When the cap or the
deadline cuts the rerank short, the result is not certified:
`confidence` is `None`. Certification saves work only when the nearest
neighbours stand clear of the rest.

`deadline_ms=` stops the code scan once the time is up. The clock is
checked between 32-vector blocks during the scan and between batches of
256 during the rescoring that follows. You get the best results found so far and `completeness`, the
fraction of vectors scanned. Both options work only on RaBitQ databases;
other modes raise `ValueError`. Details:
[ADR 0017](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0017-certified-deadline-search.md).

```python
import random
import srvdb

rng = random.Random(3)
vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(2000)]
ids = [f"doc-{i}" for i in range(len(vectors))]
db = srvdb.create("./certified-db", dimension=128, index="rabitq", bits=4)
db.add(ids, vectors, ["{}"] * len(ids))

out = db.search(vectors[7], k=1, certify=True)
print(out[0][0], out.certified, out.reranked, out.confidence)
# doc-7 1 0 0.99
late = db.search(vectors[7], k=1, deadline_ms=0)
print(len(late), late.completeness)
# 0 0.0
```

## Try it in SrvDB

```python
import random
import srvdb

rng = random.Random(7)
vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(2000)]
ids = [f"doc-{i}" for i in range(len(vectors))]

db = srvdb.create("./rabitq-db", dimension=128, index="rabitq", bits=2)
db.add(ids, vectors, ["{}"] * len(ids))
db.save()

top = db.search(vectors[42], k=3)   # 3 x 20 candidates, reranked exactly
print(top[0][0], round(top[0][1], 3))
# doc-42 1.0
db.set_rabitq_rerank(0)             # estimates only, no rerank
print(len(db.search(vectors[42], k=3)))
# 3
```

## Trade-offs

Sizes follow from the record layout: `4 × D` bytes in `vectors.bin`, plus
`4 + ceil(D × b / 8)` bytes in `rabitq_codes.bin`. At `D = 768`:

| Mode | On disk per vector | Read by the scan per vector | Scan bytes vs flat |
|------|-------------------:|----------------------------:|-------------------:|
| `flat` | 3072 | 3072 | 1× |
| `rabitq1` | 3172 | 100 | 30.7× fewer |
| `rabitq2` | 3268 | 196 | 15.7× fewer |
| `rabitq4` | 3460 | 388 | 7.9× fewer |

The rerank step then reads `k × rerank` full vectors: 100 vectors
(about 300 KB) for `k = 10` at the 4-bit default.

- **Recall.** With rerank on, results are exact cosine scores for the
  candidates found; recall depends on whether the true neighbours make the
  `k × rerank` cut. Tightly clustered data needs a larger `rerank`.
  Measure against [flat](/srvdb/concepts/flat/) on your own data.
- **Speed.** Fewer bits make the scan read fewer bytes but need a larger
  rerank. The scan is still linear in the number of vectors.
  RaBitQ fast-scan timings exist only as **preliminary** measurements from a
  loaded machine; they are not quoted here. See `docs/benchmarks.md`
  ("RaBitQ fast-scan (ADR 0012): preliminary") and issue #54 for the
  idle-machine rerun, and the [benchmarks page](/srvdb/benchmarks/).
- **Memory.** The interleaved copy (`n × ceil(D × b / 8)` bytes, plus up
  to 31 vectors of padding) is always resident, and the operating system
  may also cache pages of the code file, so codes can be in memory twice.
  If the copy cannot be allocated, open or add fails with `OSError` (Rust
  `Error::Io`, kind `OutOfMemory`) instead of aborting.
- **Open cost.** Each open regenerates the rotation (O(D³) work) and
  makes one pass over the codes.

## When to choose it

- [Flat](/srvdb/concepts/flat/) is too slow, you want near-exact results,
  and you do not want to build or tune a graph.
- You cannot train a quantizer: data arrives a few vectors at a time.
- **Not** to save disk: choose [SQ8](/srvdb/concepts/sq8/) or
  [PQ](/srvdb/concepts/pq/) for that. For sub-linear search on large
  collections, choose [HNSW](/srvdb/concepts/hnsw/).

Compare every mode on one page in [Compare index modes](/srvdb/concepts/comparison/).
API: [`srvdb.create`](/srvdb/reference/python/create/),
[`db.set_rabitq_rerank`](/srvdb/reference/python/db-set-rabitq-rerank/),
[`db.search`](/srvdb/reference/python/db-search/) (`certify`, `deadline_ms`),
and the [on-disk files](/srvdb/reference/files/).

## FAQ

### Does RaBitQ in SrvDB need training?

No. The rotation comes from a fixed 64-bit seed and each vector is encoded
on its own, so the first vector can be added straight away.

### Are RaBitQ search scores exact in SrvDB?

With rerank on, which is the default, the returned scores are exact cosine
similarities computed from the full vectors. With rerank set to 0 they are
estimates from the codes.

### Does RaBitQ make a SrvDB database smaller?

No. Full vectors are kept for rerank, so a RaBitQ database is slightly
larger than a flat one. Use SQ8 or PQ when disk space is the limit.

### What happens if the RaBitQ code files are damaged?

The codes are derived data. If they are missing, fail validation or were
made with other settings, SrvDB rebuilds them from `vectors.bin` when the
database opens.

## Related concepts

:::note[Related concepts]
- [Flat (exact) search](/srvdb/concepts/flat/): the exact scoring that rerank uses.
- [SQ8](/srvdb/concepts/sq8/) and [PQ](/srvdb/concepts/pq/): compression that replaces the full vectors.
- [HNSW](/srvdb/concepts/hnsw/): sub-linear graph search.
- [Compare index modes](/srvdb/concepts/comparison/): every mode side by side.
:::

**Next:** [Compare index modes](/srvdb/concepts/comparison/)

## Papers and further reading

Everything SrvDB does is explained above; these papers are for deeper reading.

- Jianyang Gao, Cheng Long (2024). *RaBitQ: Quantizing High-Dimensional
  Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor
  Search.* SIGMOD 2024. [arXiv:2405.12497](https://arxiv.org/abs/2405.12497)
- Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, Raymond
  Chi-Wing Wong (2025). *Practical and Asymptotically Optimal Quantization
  of High-Dimensional Vectors in Euclidean Space for Approximate Nearest
  Neighbor Search* (extended RaBitQ, multi-bit codes). SIGMOD 2025.
  [arXiv:2409.09913](https://arxiv.org/abs/2409.09913)
- Fabien André, Anne-Marie Kermarrec, Nicolas Le Scouarnec (2015).
  *Cache locality is not enough: High-performance nearest neighbor search
  with product quantization fast scan.* VLDB 2015. The 4-bit
  lookup-table shuffle that SrvDB's fast-scan builds on.
