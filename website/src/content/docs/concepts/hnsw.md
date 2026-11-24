---
title: What is an HNSW index?
description: How SrvDB's HNSW mode builds a layered neighbour graph, descends it at search time, saves it with persist(), and supports filtered search - with m, ef settings and memory per vector.
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: '{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[{"@type":"Question","name":"Is HNSW search exact in SrvDB?","acceptedAnswer":{"@type":"Answer","text":"No, it is approximate: the graph walk can miss a true neighbour. The scores it returns are true cosine similarities, and raising ef_search raises recall."}},{"@type":"Question","name":"What happens if the HNSW graph file is lost?","acceptedAnswer":{"@type":"Answer","text":"The graph is derived data. SrvDB rebuilds it from vectors.bin when hnsw.graph is missing or damaged."}},{"@type":"Question","name":"Which HNSW settings does SrvDB use by default?","acceptedAnswer":{"@type":"Answer","text":"m = 16 links per node (32 on layer 0), ef_construction = 200 and ef_search = 50."}}]}'
---

HNSW (Hierarchical Navigable Small World) is a graph index: each vector is
a node linked to its near neighbours, on several layers of decreasing
density. A search starts at the sparse top layer and walks greedily
downwards, so it scores a small part of the collection instead of all of it.

## How it works

<figure class="concept-diagram">
<svg viewBox="0 0 720 240" role="img" aria-labelledby="hnsw-t">
<title id="hnsw-t">HNSW descent: greedy steps on sparse upper layers, then a wider ef search on layer 0</title>
<defs><marker id="hnsw-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<text x="10" y="40" class="dg-muted">layer 2</text>
<text x="10" y="120" class="dg-muted">layer 1</text>
<text x="10" y="200" class="dg-muted">layer 0</text>
<path class="dg-line" d="M140,36 L420,36"/>
<circle class="dg-dot-hi" cx="140" cy="36" r="6"/><circle class="dg-dot" cx="420" cy="36" r="5"/>
<text x="140" y="20" text-anchor="middle" class="dg-muted">entry point</text>
<path class="dg-line" d="M120,116 L250,106 L420,116 L560,104 M250,106 L340,130"/>
<circle class="dg-dot" cx="120" cy="116" r="5"/><circle class="dg-dot" cx="250" cy="106" r="5"/><circle class="dg-dot-hi" cx="420" cy="116" r="6"/><circle class="dg-dot" cx="560" cy="104" r="5"/><circle class="dg-dot" cx="340" cy="130" r="5"/>
<path class="dg-line" d="M100,196 L170,186 L250,200 L320,184 L390,198 L460,188 L530,202 L600,190 L670,198 M320,184 L460,188 M390,198 L460,212 L530,202"/>
<circle class="dg-dot" cx="100" cy="196" r="4"/><circle class="dg-dot" cx="170" cy="186" r="4"/><circle class="dg-dot" cx="250" cy="200" r="4"/><circle class="dg-dot" cx="320" cy="184" r="4"/><circle class="dg-dot-hi" cx="390" cy="198" r="4"/><circle class="dg-dot-hi" cx="460" cy="188" r="6"/><circle class="dg-dot-hi" cx="460" cy="212" r="4"/><circle class="dg-dot-hi" cx="530" cy="202" r="4"/><circle class="dg-dot" cx="600" cy="190" r="4"/><circle class="dg-dot" cx="670" cy="198" r="4"/>
<path class="dg-arrow" d="M150,40 C260,50 380,60 414,108" marker-end="url(#hnsw-a)"/>
<path class="dg-arrow" d="M424,124 C440,150 452,165 458,180" marker-end="url(#hnsw-a)"/>
<text x="560" y="150" class="dg-muted">query is near here</text>
<text x="480" y="232" class="dg-muted">ef best candidates kept → top k</text>
</svg>
<figcaption>Upper layers are sparse and give long jumps; layer 0 holds every vector and is searched with a candidate list of size ef.</figcaption>
</figure>

1. **Levels.** A new node gets a random top layer; the chance of reaching
   each higher layer falls by a factor of `m` (level multiplier `1 / ln m`).
   So almost every node lives only on layer 0, and a few reach the top.
2. **Insert.** The node is placed by searching the graph with a candidate
   list of `ef_construction`. Neighbours are chosen with the diversity
   heuristic: a candidate is kept only if it is closer to the new node
   than to every neighbour already kept. This keeps long links between
   clusters, so clustered data does not split into islands (issue #28).
   Each node keeps up to `m` links per upper layer and `2 × m` on layer 0.
3. **Search.** From the entry point the walk moves greedily to the
   neighbour closest to the query on each upper layer, then runs a
   best-first search on layer 0 that keeps `ef_search` candidates (at
   least `k`). Scores are exact cosine similarities from `vectors.bin`.
4. **Persist.** `persist()` writes `hnsw.graph` atomically (temp file,
   fsync, rename). The graph is derived data: if it is missing or damaged
   it is rebuilt from the vectors on open.
5. **Filtered search (Rust).** `search_filtered` walks the graph and, when
   it reaches neighbours the filter rejects, looks at their neighbours
   (ACORN-1 style two-hop expansion). When few ids match it scores them
   exactly instead. See [ADR 0010](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0010-filtered-search.md)
   and [Using SrvDB from Rust](/srvdb/guides/rust/).

Durable vector storage, quantized modes and filtered HNSW in one embedded
library, with no server: that combination is what SrvDB is built for.

## Try it in SrvDB

```python
import random
import srvdb

rng = random.Random(7)
vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(1000)]
ids = [f"doc-{i}" for i in range(len(vectors))]

db = srvdb.Database.new_hnsw("./hnsw-db", dimension=128, m=16, ef_construction=200, ef_search=50)
db.add(ids, vectors, ["{}"] * len(ids))
db.set_ef_search(120)   # wider search: higher recall, slower
db.persist()            # hnsw.graph is saved and reloaded on open
print(len(db.search(vectors[42], k=5)))
# 5
```

## Trade-offs

- **Recall.** Approximate. Raise `ef_search` (search time) or `m` and
  `ef_construction` (build time) for higher recall. Measure against
  [flat](/srvdb/concepts/flat/) on your own data.
- **Speed.** A search visits a small part of the graph instead of all
  `n` vectors. Inserts cost more than in flat mode, because each one runs
  a graph search.
- **Bytes per vector.** The full vectors (`4 × D`) plus the links. With
  `m = 16`, layer 0 holds up to 32 neighbour ids per node; upper layers
  add on average about one more list of up to 16 for every 15 nodes. In
  memory each id is 8 bytes, so up to about 256 bytes of layer-0 links per
  node; on disk `hnsw.graph` uses variable-length (postcard) encoding.
- **Limits.** `m` is 2 to 256; `ef_construction` and `ef_search` are 1 to
  10000. Values outside raise `ValueError`.

## When to choose it

- Flat search is too slow and approximate results are acceptable.
- Data is added gradually: no training step, unlike [IVF](/srvdb/concepts/ivf/).
- You need filtered search over metadata from Rust.
- Not when disk is the limit: HNSW stores full vectors plus links. Then
  see [SQ8](/srvdb/concepts/sq8/) or [PQ](/srvdb/concepts/pq/).

## FAQ

### Is HNSW search exact in SrvDB?

No, it is approximate: the graph walk can miss a true neighbour. The scores
it returns are true cosine similarities, and raising `ef_search` raises
recall.

### What happens if the HNSW graph file is lost?

The graph is derived data. SrvDB rebuilds it from `vectors.bin` when
`hnsw.graph` is missing or damaged.

### Which HNSW settings does SrvDB use by default?

`m = 16` links per node (32 on layer 0), `ef_construction = 200` and
`ef_search = 50`.

## Related concepts

:::note[Related concepts]
- [IVF](/srvdb/concepts/ivf/): partitions, each with its own HNSW graph.
- [Flat (exact) search](/srvdb/concepts/flat/): the ground truth for recall.
- [Compare index modes](/srvdb/concepts/comparison/).
- Reference: [`Database.new_hnsw`](/srvdb/reference/python/database-new-hnsw/), [`Database.set_ef_search`](/srvdb/reference/python/database-set-ef-search/).
:::

**Next:** [What is an IVF index?](/srvdb/concepts/ivf/)

## Papers and further reading

- Yu. A. Malkov, D. A. Yashunin (2016, journal version 2018). *Efficient
  and robust approximate nearest neighbor search using Hierarchical
  Navigable Small World graphs.* IEEE TPAMI.
  [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- Liana Patel, Peter Kraft, Carlos Guestrin, Matei Zaharia (2024).
  *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings
  and Structured Data.* SIGMOD 2024.
  [arXiv:2403.04871](https://arxiv.org/abs/2403.04871)
