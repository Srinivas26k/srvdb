---
title: What is SQ8 scalar quantization?
description: How SrvDB's SQ8 mode maps each dimension's min-max range to one byte (0-255), scans the codes with cosine similarity, and stores vectors 4 times smaller than flat.
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: '{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[{"@type":"Question","name":"How much smaller is SQ8 than flat in SrvDB?","acceptedAnswer":{"@type":"Answer","text":"4 times: each dimension is stored as 1 byte instead of a 4-byte float, so a 768-dimensional vector takes 768 bytes instead of 3072."}},{"@type":"Question","name":"Does SQ8 keep the original vectors?","acceptedAnswer":{"@type":"Answer","text":"No. Only the 8-bit codes are stored, so search scores and returned vectors are approximations of the originals."}},{"@type":"Question","name":"What is SQ8 trained on?","acceptedAnswer":{"@type":"Answer","text":"On the per-dimension minimum and maximum of the training vectors: the first batch added, or the sample passed to new_scalar_quantized. Later values outside that range are clamped."}}]}'
---

SQ8 (8-bit scalar quantization) stores every dimension of a vector as one
byte: the value's position between that dimension's minimum and maximum,
scaled to 0-255. It makes a SrvDB database 4 times smaller than flat with
one simple, fast transformation, at the cost of approximate scores.

## How it works

<figure class="concept-diagram">
<svg viewBox="0 0 720 190" role="img" aria-labelledby="sq8-t">
<title id="sq8-t">SQ8: a dimension's range from min to max is split into 256 levels; a value maps to the level below it</title>
<defs><marker id="sq8-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<text x="10" y="24" class="dg-muted">dimension i, learned from training vectors</text>
<path class="dg-line" d="M60,70 H660"/>
<path class="dg-line" d="M60,58 V82 M660,58 V82"/>
<path class="dg-line" d="M135,64 V76 M210,64 V76 M285,64 V76 M360,64 V76 M435,64 V76 M510,64 V76 M585,64 V76"/>
<text x="60" y="102" text-anchor="middle">min_i</text>
<text x="660" y="102" text-anchor="middle">max_i</text>
<text x="60" y="50" text-anchor="middle" class="dg-muted">0</text>
<text x="660" y="50" text-anchor="middle" class="dg-muted">255</text>
<circle class="dg-dot-hi" cx="402" cy="70" r="6"/>
<text x="402" y="50" text-anchor="middle">x_i</text>
<rect class="dg-hi" x="250" y="130" width="300" height="40" rx="6"/>
<text x="400" y="155" text-anchor="middle">code = floor((x_i - min_i) / range × 255)</text>
<path class="dg-arrow" d="M402,78 V126" marker-end="url(#sq8-a)"/>
<text x="10" y="155" class="dg-muted">1 byte per dimension</text>
</svg>
<figcaption>Each dimension has its own range. Values outside it are clamped to 0 or 255; a dimension with no range is stored as 0.</figcaption>
</figure>

1. **Train.** SrvDB records the minimum and maximum of every dimension
   over the training vectors: the sample passed to
   `Database.new_scalar_quantized`, or the first batch added to a database
   opened with `mode="sq8"`. The ranges are saved atomically in
   `scalar_quantizer.bin`.
2. **Encode.** Value `x_i` becomes
   `floor(clamp((x_i - min_i) / (max_i - min_i) × 255, 0, 255))`.
   Codes go to `sq8_vectors.bin`, one `D`-byte record per vector, with the
   same versioned header and CRC32C sidecar as `vectors.bin`.
3. **Search.** The scan scores the query (full f32) against every code
   vector, decoding `min_i + code × (max_i - min_i) / 255` on the fly and
   accumulating the dot product and norm in one pass, in parallel blocks
   of 256 vectors. The score is the cosine similarity between the query
   and the decoded vector.
4. **No originals.** The full vectors are not stored. A quantizer file
   cannot be retrained over existing codes, because they would no longer
   decode correctly.

## Try it in SrvDB

```python
import random
import srvdb

rng = random.Random(7)
vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(1000)]
ids = [f"doc-{i}" for i in range(len(vectors))]

db = srvdb.Database.new_scalar_quantized("./sq8-db", 128, vectors[:200])
db.add(ids, vectors, ["{}"] * len(ids))
db.persist()
print(db.search(vectors[42], k=1)[0][0])
# doc-42
```

## Trade-offs

- **Bytes per vector.** `D`: 128 at D = 128, 384 at D = 384, 768 at
  D = 768. Four times less than [flat](/srvdb/concepts/flat/).
- **Recall.** Approximate. The step per dimension is `range / 255`, so the
  error depends on how wide the ranges are. Outliers in the training
  sample widen a range and coarsen every code in that dimension; values
  beyond the trained range are clamped. Measure recall against flat.
- **Scores.** Cosine with the decoded vector, not with the original.
- **Speed.** Still a linear scan, but it reads a quarter of the bytes.
- **Training.** Only a min/max pass: cheap, but the sample must be
  representative of later data.

## When to choose it

- Disk or memory is the limit and you can accept approximate scores.
- Your embeddings are stable in range (most text embedding models are).
- When you need exact scores with a faster scan, choose
  [RaBitQ](/srvdb/concepts/rabitq/); for more compression, choose
  [PQ](/srvdb/concepts/pq/).

## FAQ

### How much smaller is SQ8 than flat in SrvDB?

4 times: each dimension is stored as 1 byte instead of a 4-byte float, so a
768-dimensional vector takes 768 bytes instead of 3072.

### Does SQ8 keep the original vectors?

No. Only the 8-bit codes are stored, so search scores and returned vectors
are approximations of the originals.

### What is SQ8 trained on?

On the per-dimension minimum and maximum of the training vectors: the first
batch added, or the sample passed to `new_scalar_quantized`. Later values
outside that range are clamped.

## Related concepts

:::note[Related concepts]
- [PQ](/srvdb/concepts/pq/): more compression with learned codebooks.
- [RaBitQ](/srvdb/concepts/rabitq/): low-bit codes that keep exact scores via rerank.
- [Compare index modes](/srvdb/concepts/comparison/).
- Reference: [`Database.new_scalar_quantized`](/srvdb/reference/python/database-new-scalar-quantized/), [on-disk files](/srvdb/reference/files/).
:::

**Next:** [What is product quantization?](/srvdb/concepts/pq/)

## Papers and further reading

Per-dimension min-max scalar quantization is a standard technique rather
than the subject of one paper; the [PQ page](/srvdb/concepts/pq/) lists
the product-quantization papers that compare against it.
