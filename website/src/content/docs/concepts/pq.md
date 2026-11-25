---
title: What is product quantization (PQ)?
description: How SrvDB's PQ mode splits each vector into sub-vectors of 8 (or 4) dimensions, stores one learned centroid index per sub-vector, and scans with lookup tables - up to 32 times smaller than flat.
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: '{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[{"@type":"Question","name":"How small are PQ codes in SrvDB?","acceptedAnswer":{"@type":"Answer","text":"One byte per sub-vector: dimension / 8 bytes when the dimension is divisible by 8, otherwise dimension / 4. At dimension 768 that is 96 bytes, 32 times less than flat."}},{"@type":"Question","name":"How much data does PQ need for training?","acceptedAnswer":{"@type":"Answer","text":"Each sub-space learns 256 centroids with k-means. A database opened with mode pq trains on the first batch added, which must hold at least 256 vectors."}},{"@type":"Question","name":"Are PQ scores exact?","acceptedAnswer":{"@type":"Answer","text":"No. The score is the average, over sub-spaces, of the cosine between the query sub-vector and the stored centroid. The full vectors are not kept."}}]}'
---

Product quantization (PQ) cuts each vector into short sub-vectors and
replaces every sub-vector by the index of its nearest centroid in a
learned 256-entry codebook, one byte each. In SrvDB that stores a
768-dimensional vector in 96 bytes, 32 times less than flat, and searches
it with per-query lookup tables.

:::caution[Known issue #95]
PQ recall is low in this release
([#95](https://github.com/Srinivas26k/srvdb/issues/95), targeted at 0.4.1).
For compressed storage, use [RaBitQ](/srvdb/concepts/rabitq/)
(`index="rabitq"`) instead.
:::

## How it works

<figure class="concept-diagram">
<svg viewBox="0 0 720 210" role="img" aria-labelledby="pq-t">
<title id="pq-t">PQ: a vector is split into sub-vectors; each is replaced by the index of its nearest of 256 centroids</title>
<defs><marker id="pq-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<text x="10" y="22" class="dg-muted">vector, D dimensions, split into M = D / 8 sub-vectors</text>
<rect class="dg-box" x="10" y="32" width="120" height="32"/><rect class="dg-box" x="130" y="32" width="120" height="32"/><rect class="dg-box" x="250" y="32" width="120" height="32"/><rect class="dg-box" x="370" y="32" width="120" height="32"/>
<text x="70" y="53" text-anchor="middle">8 floats</text><text x="190" y="53" text-anchor="middle">8 floats</text><text x="310" y="53" text-anchor="middle">8 floats</text><text x="430" y="53" text-anchor="middle">...</text>
<path class="dg-arrow" d="M70,64 V100" marker-end="url(#pq-a)"/><path class="dg-arrow" d="M190,64 V100" marker-end="url(#pq-a)"/><path class="dg-arrow" d="M310,64 V100" marker-end="url(#pq-a)"/>
<rect class="dg-box" x="20" y="104" width="100" height="40" rx="4"/><rect class="dg-box" x="140" y="104" width="100" height="40" rx="4"/><rect class="dg-box" x="260" y="104" width="100" height="40" rx="4"/>
<text x="70" y="128" text-anchor="middle" class="dg-muted">codebook 1</text><text x="190" y="128" text-anchor="middle" class="dg-muted">codebook 2</text><text x="310" y="128" text-anchor="middle" class="dg-muted">codebook 3</text>
<path class="dg-arrow" d="M70,144 V170" marker-end="url(#pq-a)"/><path class="dg-arrow" d="M190,144 V170" marker-end="url(#pq-a)"/><path class="dg-arrow" d="M310,144 V170" marker-end="url(#pq-a)"/>
<rect class="dg-hi" x="40" y="174" width="60" height="28"/><rect class="dg-hi" x="160" y="174" width="60" height="28"/><rect class="dg-hi" x="280" y="174" width="60" height="28"/>
<text x="70" y="193" text-anchor="middle">17</text><text x="190" y="193" text-anchor="middle">203</text><text x="310" y="193" text-anchor="middle">88</text>
<text x="520" y="128" class="dg-muted">256 centroids each,</text>
<text x="520" y="146" class="dg-muted">learned by k-means</text>
<text x="400" y="193" class="dg-muted">1 byte per sub-vector</text>
</svg>
<figcaption>The code is M bytes. At search time each sub-space gets a 256-entry table of query-to-centroid scores, so scoring a vector is M table lookups.</figcaption>
</figure>

1. **Split.** SrvDB uses `M = D / 8` sub-spaces when `D` is divisible by
   8, otherwise `D / 4`. Other dimensions are refused with `ValueError`.
2. **Train.** Each sub-space learns 256 centroids with k-means (20
   iterations), trained in parallel. A database created with `index="pq"` and no `training`
   trains on its first batch, which must hold at least 256 vectors;
   `training=` trains on the sample you pass. The
   codebooks are saved in `codebooks.bin`.
3. **Encode.** Each sub-vector is replaced by the index of the centroid
   with the highest cosine similarity. Codes go to
   `quantized_vectors.bin`, `M` bytes per vector.
4. **Search (asymmetric distance).** The query stays in full precision.
   For each sub-space SrvDB computes its similarity to all 256 centroids
   once, giving `M` tables. A stored vector's score is then the average of
   `M` table lookups, scanned in parallel blocks of 256 vectors.

## Try it in SrvDB

```python
import random
import srvdb

rng = random.Random(7)
vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(1000)]
ids = [f"doc-{i}" for i in range(len(vectors))]

db = srvdb.create("./pq-db", dimension=128, index="pq", training=vectors[:300])
db.add(ids, vectors, ["{}"] * len(ids))
db.save()
print(len(db.search(vectors[42], k=5)))
# 5
```

## Trade-offs

- **Bytes per vector.** `D / 8`: 16 at D = 128, 48 at D = 384, 96 at
  D = 768 (32 times less than flat). With `D / 4` sub-spaces, 16 times
  less.
- **Recall.** Approximate and bounded by the codebooks: two vectors with
  the same codes are indistinguishable. Tightly clustered embeddings can
  lose a lot of accuracy. Measure against [flat](/srvdb/concepts/flat/).
- **Scores.** An average of per-sub-space cosines, not the true cosine.
  The full vectors are not stored, so there is no exact rerank.
- **Speed.** A linear scan of `M` lookups per vector; table building is
  `256 × D` work per query.
- **Training.** k-means over every sub-space; the cost grows with the
  sample size. Codes depend on the codebooks, so they cannot be retrained
  in place.

## When to choose it

- The collection must be as small as possible on disk and in memory, and
  approximate ranking is acceptable.
- You can supply a representative training sample of 256 vectors or more.
- If you need exact scores, choose [RaBitQ](/srvdb/concepts/rabitq/) (no
  training, full vectors kept); for 4× compression with a simpler model,
  choose [SQ8](/srvdb/concepts/sq8/).

## FAQ

### How small are PQ codes in SrvDB?

One byte per sub-vector: dimension / 8 bytes when the dimension is
divisible by 8, otherwise dimension / 4. At dimension 768 that is 96 bytes,
32 times less than flat.

### How much data does PQ need for training?

Each sub-space learns 256 centroids with k-means. A database opened with
`index="pq"` without `training` trains on the first batch added, which must hold at least 256
vectors.

### Are PQ scores exact?

No. The score is the average, over sub-spaces, of the cosine between the
query sub-vector and the stored centroid. The full vectors are not kept.

## Related concepts

:::note[Related concepts]
- [SQ8](/srvdb/concepts/sq8/): simpler 4× compression.
- [RaBitQ](/srvdb/concepts/rabitq/): training-free codes, exact rerank.
- [IVF](/srvdb/concepts/ivf/): the same k-means idea, applied to whole vectors.
- Reference: [`srvdb.create`](/srvdb/reference/python/create/).
:::

**Next:** [What is RaBitQ quantization?](/srvdb/concepts/rabitq/)

## Papers and further reading

- Hervé Jégou, Matthijs Douze, Cordelia Schmid (2011). *Product
  Quantization for Nearest Neighbor Search.* IEEE Transactions on Pattern
  Analysis and Machine Intelligence 33(1).
  [doi:10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
- Fabien André, Anne-Marie Kermarrec, Nicolas Le Scouarnec (2015).
  *Cache locality is not enough: High-performance nearest neighbor search
  with product quantization fast scan.* VLDB 2015. The lookup-table
  shuffle that SrvDB's [RaBitQ fast-scan](/srvdb/concepts/rabitq/) uses.
