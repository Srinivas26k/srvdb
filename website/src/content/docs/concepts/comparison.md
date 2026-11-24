---
title: Which SrvDB index mode should I use?
description: Flat, HNSW, IVF, SQ8, PQ and RaBitQ side by side in SrvDB - how each searches, bytes per vector computed from the file formats, training needs, score exactness and when to pick it.
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: '{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[{"@type":"Question","name":"Which SrvDB mode should I start with?","acceptedAnswer":{"@type":"Answer","text":"Start with flat. It is exact, needs no training and is the baseline for measuring every other mode. Switch only when search is too slow or files are too large."}},{"@type":"Question","name":"Which SrvDB modes return exact cosine scores?","acceptedAnswer":{"@type":"Answer","text":"Flat, HNSW, IVF and RaBitQ with rerank on score candidates against the full vectors. SQ8 and PQ score against compressed codes, and RaBitQ with rerank 0 returns estimates."}},{"@type":"Question","name":"Which SrvDB modes need training?","acceptedAnswer":{"@type":"Answer","text":"IVF (train_ivf), SQ8 (min and max per dimension) and PQ (k-means codebooks). Flat, HNSW and RaBitQ need none."}}]}'
---

Start with flat, which is exact, and change mode only when search is too
slow or files are too large. SrvDB gives you six ways to search the same
durable, checksummed vector store, all in one embedded library with no
server: pick by recall, speed and bytes per vector.

## All modes at a glance

Bytes per vector come from the record layouts in `crates/srvdb/src/storage`
and exclude metadata and the small CRC sidecars. `D` is the dimension.

| Mode | How it searches | Scores | Bytes per vector | At D = 768 | Training |
|------|-----------------|--------|------------------|-----------:|----------|
| [`flat`](/srvdb/concepts/flat/) | Scan of every vector | Exact | `4D` | 3072 | None |
| [`hnsw`](/srvdb/concepts/hnsw/) | Layered graph walk | Exact cosine, approximate ranking | `4D` + links (≤ `2m` ids on layer 0) | 3072 + links | None |
| [`ivf`](/srvdb/concepts/ivf/) | Probe `nprobe` partitions, HNSW in each | Exact cosine, approximate ranking | `4D` + links (+ centroids per index) | 3072 + links | `train_ivf()` |
| [`sq8`](/srvdb/concepts/sq8/) | Scan of 8-bit codes | Approximate | `D` | 768 | First batch or sample |
| [`pq`](/srvdb/concepts/pq/) | Scan with lookup tables | Approximate | `D / 8` (or `D / 4`) | 96 | First batch ≥ 256 or sample |
| [`rabitq1`](/srvdb/concepts/rabitq/) | Code scan, then rerank | Exact with rerank | `4D + 4 + ceil(D/8)` | 3172 | None |
| [`rabitq2`](/srvdb/concepts/rabitq/) | Code scan, then rerank | Exact with rerank | `4D + 4 + ceil(D/4)` | 3268 | None |
| [`rabitq4`](/srvdb/concepts/rabitq/) | Code scan, then rerank | Exact with rerank | `4D + 4 + ceil(D/2)` | 3460 | None |

"Exact cosine, approximate ranking" means the returned scores are true
cosine similarities, but a true neighbour can be missed. RaBitQ's scan
reads only the codes: 100, 196 or 388 bytes per vector at D = 768.

## Decision guide

<figure class="concept-diagram">
<svg viewBox="0 0 720 200" role="img" aria-labelledby="cmp-t">
<title id="cmp-t">Decision guide: start flat; if too slow choose RaBitQ or HNSW; if too large choose SQ8 or PQ</title>
<defs><marker id="cmp-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<rect class="dg-hi" x="290" y="10" width="140" height="40" rx="6"/>
<text x="360" y="35" text-anchor="middle">start: flat</text>
<path class="dg-arrow" d="M330,50 L190,92" marker-end="url(#cmp-a)"/>
<path class="dg-arrow" d="M390,50 L530,92" marker-end="url(#cmp-a)"/>
<text x="200" y="70" class="dg-muted">too slow?</text>
<text x="470" y="70" class="dg-muted">too large?</text>
<rect class="dg-box" x="40" y="96" width="140" height="40" rx="6"/><text x="110" y="121" text-anchor="middle">rabitq</text>
<rect class="dg-box" x="200" y="96" width="140" height="40" rx="6"/><text x="270" y="121" text-anchor="middle">hnsw / ivf</text>
<rect class="dg-box" x="390" y="96" width="140" height="40" rx="6"/><text x="460" y="121" text-anchor="middle">sq8</text>
<rect class="dg-box" x="550" y="96" width="140" height="40" rx="6"/><text x="620" y="121" text-anchor="middle">pq</text>
<text x="110" y="158" text-anchor="middle" class="dg-muted">exact scores,</text><text x="110" y="174" text-anchor="middle" class="dg-muted">no training</text>
<text x="270" y="158" text-anchor="middle" class="dg-muted">sub-linear,</text><text x="270" y="174" text-anchor="middle" class="dg-muted">approximate</text>
<text x="460" y="158" text-anchor="middle" class="dg-muted">4× smaller</text>
<text x="620" y="158" text-anchor="middle" class="dg-muted">up to 32× smaller</text>
</svg>
<figcaption>Whatever you pick, measure recall against flat on a sample of your own data.</figcaption>
</figure>

1. **Exact results, small or medium collection:** [flat](/srvdb/concepts/flat/).
2. **Scan too slow, want flat-like results, no training:**
   [RaBitQ](/srvdb/concepts/rabitq/).
3. **Large collection, sub-linear search:** [HNSW](/srvdb/concepts/hnsw/);
   [IVF](/srvdb/concepts/ivf/) when the data is present before training.
4. **Disk or memory is the limit:** [SQ8](/srvdb/concepts/sq8/), then
   [PQ](/srvdb/concepts/pq/).

Performance depends on your hardware and data. Published, reproducible
measurements, with hardware, OS, dataset, commit and command, are on the
[benchmarks page](/srvdb/benchmarks/). RaBitQ fast-scan timings are
preliminary and not quoted here.

## Try it in SrvDB

```python
import random
import srvdb

rng = random.Random(7)
vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(500)]
ids = [f"doc-{i}" for i in range(len(vectors))]

for index in ["flat", "hnsw", "sq8", "rabitq"]:
    db = srvdb.create(f"./db-{index}", dimension=128, index=index)
    db.add(ids, vectors, ["{}"] * len(ids))
    db.save()
    print(index, db.search(vectors[42], k=1)[0][0])
```

[Choosing an index mode](/srvdb/guides/index-modes/) has a `recall_at_k`
helper for comparing modes on your own data.

## FAQ

### Which SrvDB mode should I start with?

Start with flat. It is exact, needs no training and is the baseline for
measuring every other mode. Switch only when search is too slow or files
are too large.

### Which SrvDB modes return exact cosine scores?

Flat, HNSW, IVF and RaBitQ with rerank on score candidates against the full
vectors. SQ8 and PQ score against compressed codes, and RaBitQ with rerank
0 returns estimates.

### Which SrvDB modes need training?

IVF (`train_ivf`), SQ8 (min and max per dimension) and PQ (k-means
codebooks). Flat, HNSW and RaBitQ need none.

## Related concepts

:::note[Related concepts]
- [Flat](/srvdb/concepts/flat/), [HNSW](/srvdb/concepts/hnsw/), [IVF](/srvdb/concepts/ivf/), [SQ8](/srvdb/concepts/sq8/), [PQ](/srvdb/concepts/pq/), [RaBitQ](/srvdb/concepts/rabitq/).
- Guides: [Choosing an index mode](/srvdb/guides/index-modes/), [Durability and save()](/srvdb/guides/durability/).
- Reference: [Python API](/srvdb/reference/python/), [on-disk files](/srvdb/reference/files/).
:::

**Next:** [What is flat (exact) vector search?](/srvdb/concepts/flat/)

## Papers and further reading

Each concept page ends with the papers behind its method.
