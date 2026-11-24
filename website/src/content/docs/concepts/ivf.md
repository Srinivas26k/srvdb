---
title: What is an IVF index?
description: How SrvDB's IVF mode clusters vectors with spherical k-means into nlist partitions, probes the nprobe closest at search time, and searches each with its own HNSW graph.
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: '{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[{"@type":"Question","name":"Does IVF in SrvDB need training?","acceptedAnswer":{"@type":"Answer","text":"Yes. Call train_ivf() after adding data; it clusters up to the first 10000 stored vectors. Until then, IVF mode searches exactly, like flat."}},{"@type":"Question","name":"What do nlist and nprobe mean?","acceptedAnswer":{"@type":"Answer","text":"nlist is the number of partitions (centroids) and nprobe is how many of the closest partitions a search looks in. The defaults are 100 and 10."}},{"@type":"Question","name":"Is the trained IVF index saved?","acceptedAnswer":{"@type":"Answer","text":"Yes. persist() saves ivf.index and it is reloaded on open. If it does not cover the committed vectors, SrvDB retrains."}}]}'
---

IVF (inverted file) splits the collection into `nlist` partitions around
learned centroids and, at query time, searches only the `nprobe`
partitions whose centroids are closest to the query. In SrvDB each
partition has its own [HNSW](/srvdb/concepts/hnsw/) graph, so the search
is coarse (pick partitions) then fine (walk small graphs).

## How it works

<figure class="concept-diagram">
<svg viewBox="0 0 720 250" role="img" aria-labelledby="ivf-t">
<title id="ivf-t">IVF: centroids divide space into cells; the query probes the nprobe nearest cells</title>
<rect class="dg-box" x="10" y="10" width="460" height="230" rx="6"/>
<path class="dg-line" d="M170,10 L190,120 L120,240 M190,120 L320,140 L330,10 M320,140 L360,240 M320,140 L470,120"/>
<path class="dg-hi" d="M190,120 L320,140 L330,10 L170,10 Z" style="opacity:0.9"/>
<path class="dg-hi" d="M190,120 L320,140 L360,240 L120,240 Z" style="opacity:0.9"/>
<circle class="dg-dot-hi" cx="250" cy="70" r="7"/><circle class="dg-dot-hi" cx="250" cy="190" r="7"/>
<circle class="dg-dot" cx="90" cy="70" r="7"/><circle class="dg-dot" cx="80" cy="190" r="7"/><circle class="dg-dot" cx="400" cy="60" r="7"/><circle class="dg-dot" cx="420" cy="190" r="7"/>
<circle class="dg-dot" cx="230" cy="50" r="3"/><circle class="dg-dot" cx="280" cy="90" r="3"/><circle class="dg-dot" cx="220" cy="210" r="3"/><circle class="dg-dot" cx="270" cy="170" r="3"/><circle class="dg-dot" cx="60" cy="100" r="3"/><circle class="dg-dot" cx="420" cy="90" r="3"/><circle class="dg-dot" cx="400" cy="210" r="3"/>
<path class="dg-arrow" d="M240,118 l8,8 m0,-8 l-8,8"/>
<text x="258" y="130">q</text>
<text x="500" y="40">● centroid</text>
<text x="500" y="70" class="dg-muted">· stored vector</text>
<rect class="dg-hi" x="500" y="88" width="16" height="14"/>
<text x="524" y="100">probed cell</text>
<text x="500" y="140" class="dg-muted">nprobe = 2 of nlist = 6:</text>
<text x="500" y="160" class="dg-muted">only shaded cells are</text>
<text x="500" y="180" class="dg-muted">searched, each with its</text>
<text x="500" y="200" class="dg-muted">own HNSW graph</text>
</svg>
<figcaption>A vector belongs to the cell of its most similar centroid; a search ranks centroids by cosine and probes the closest nprobe cells.</figcaption>
</figure>

1. **Train.** `train_ivf()` takes up to the first 10000 stored vectors,
   scales them to unit length and runs spherical k-means: each centroid is
   the normalised mean direction of its members. It stops after
   `max_iterations` (20) or when centroids move less than the tolerance.
   `nlist` is capped at the number of training vectors.
2. **Assign.** Every stored vector, and every vector added later, goes to
   the partition whose centroid has the highest cosine similarity, and is
   inserted into that partition's HNSW graph.
3. **Search.** Centroids are ranked by cosine distance to the query, the
   top `nprobe` partitions are searched in parallel, and the results are
   merged into the top `k` with exact cosine scores.
4. **Before training** IVF mode searches exactly, like flat.
5. **Persist.** `persist()` saves `ivf.index`; it is reloaded on open and
   retrained if it does not cover the committed vectors. The vectors stay
   in `vectors.bin`, so the index can always be rebuilt.

## Try it in SrvDB

```python
import random
import srvdb

rng = random.Random(7)
vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(2000)]
ids = [f"doc-{i}" for i in range(len(vectors))]

db = srvdb.open("./ivf-db", dimension=128, mode="ivf")
db.add(ids, vectors, ["{}"] * len(ids))
db.configure_ivf(nlist=20, nprobe=4)
db.train_ivf()
db.persist()
print(len(db.search(vectors[42], k=5)))
# 5
```

## Trade-offs

- **Recall.** Approximate in two ways: a true neighbour can sit in a
  partition that was not probed, and each partition's HNSW walk is itself
  approximate. Raise `nprobe` for recall; `nprobe = nlist` probes all.
- **Speed.** Ranking centroids costs `nlist` cosine computations; then
  only about `nprobe / nlist` of the data is searched.
- **Bytes per vector.** Full vectors (`4 × D`) plus HNSW links in one
  partition graph (see [HNSW](/srvdb/concepts/hnsw/)), plus
  `nlist × 4 × D` bytes of centroids for the whole index.
- **Training.** Needs data first and a call to `train_ivf()`. If the data
  drifts, centroids trained on old data fit it less well; retrain.
- **Limits.** `nlist` and `nprobe` are 1 to 65536.

## When to choose it

- Large collections whose data is mostly present before you train.
- When you want an explicit recall knob (`nprobe`) that maps to how much
  of the data is read.
- For data that arrives gradually with no training step, prefer
  [HNSW](/srvdb/concepts/hnsw/) or [RaBitQ](/srvdb/concepts/rabitq/).

## FAQ

### Does IVF in SrvDB need training?

Yes. Call `train_ivf()` after adding data; it clusters up to the first
10000 stored vectors. Until then, IVF mode searches exactly, like flat.

### What do nlist and nprobe mean?

`nlist` is the number of partitions (centroids) and `nprobe` is how many of
the closest partitions a search looks in. The defaults are 100 and 10.

### Is the trained IVF index saved?

Yes. `persist()` saves `ivf.index` and it is reloaded on open. If it does
not cover the committed vectors, SrvDB retrains.

## Related concepts

:::note[Related concepts]
- [HNSW](/srvdb/concepts/hnsw/): the graph inside every partition.
- [PQ](/srvdb/concepts/pq/): k-means again, applied to sub-vectors.
- [Compare index modes](/srvdb/concepts/comparison/).
- Reference: [`Database.configure_ivf`](/srvdb/reference/python/database-configure-ivf/), [`Database.train_ivf`](/srvdb/reference/python/database-train-ivf/).
:::

**Next:** [What is SQ8 scalar quantization?](/srvdb/concepts/sq8/)

## Papers and further reading

- Josef Sivic, Andrew Zisserman (2003). *Video Google: A Text Retrieval
  Approach to Object Matching in Videos.* ICCV 2003. The inverted file
  over learned visual words.
- Hervé Jégou, Matthijs Douze, Cordelia Schmid (2011). *Product
  Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1). Introduces
  IVFADC, the inverted file used with compressed codes.
