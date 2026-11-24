---
title: What is flat (exact) vector search?
description: How SrvDB's default flat mode scores every stored vector by cosine similarity, what it costs per vector, and when exact search is the right choice.
head:
  - tag: script
    attrs:
      type: application/ld+json
    content: '{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[{"@type":"Question","name":"Is flat search in SrvDB exact?","acceptedAnswer":{"@type":"Answer","text":"Yes. Every live vector is scored by cosine similarity against the query, so the top k results are the true nearest neighbours."}},{"@type":"Question","name":"Does flat mode need an index build or training?","acceptedAnswer":{"@type":"Answer","text":"No. Vectors are searchable as soon as add() returns; there is nothing to build, train or tune."}},{"@type":"Question","name":"How much disk does flat mode use per vector?","acceptedAnswer":{"@type":"Answer","text":"4 bytes per dimension in vectors.bin, for example 3072 bytes at dimension 768, plus metadata and a small checksum sidecar."}}]}'
---

Flat search compares the query with every stored vector and returns the
`k` with the highest cosine similarity. It is SrvDB's default mode: exact
results, no index to build, and the baseline every other mode is measured
against.

## How it works

<figure class="concept-diagram">
<svg viewBox="0 0 720 170" role="img" aria-labelledby="flat-t">
<title id="flat-t">Flat search: the query is scored against every vector in blocks of 256, and a top-k heap keeps the best</title>
<defs><marker id="flat-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<rect class="dg-hi" x="10" y="60" width="90" height="40" rx="6"/>
<text x="55" y="85" text-anchor="middle">query</text>
<text x="140" y="22" class="dg-muted">vectors.bin, scanned in blocks of 256 (parallel)</text>
<rect class="dg-box" x="140" y="32" width="120" height="30" rx="4"/><text x="200" y="52" text-anchor="middle">block 0</text>
<rect class="dg-box" x="140" y="68" width="120" height="30" rx="4"/><text x="200" y="88" text-anchor="middle">block 1</text>
<rect class="dg-box" x="140" y="104" width="120" height="30" rx="4"/><text x="200" y="124" text-anchor="middle">block ...</text>
<path class="dg-arrow" d="M100,80 H136" marker-end="url(#flat-a)"/>
<rect class="dg-box" x="320" y="60" width="170" height="40" rx="6"/>
<text x="405" y="85" text-anchor="middle">cosine per vector</text>
<path class="dg-arrow" d="M260,83 H316" marker-end="url(#flat-a)"/>
<rect class="dg-hi" x="550" y="60" width="160" height="40" rx="6"/>
<text x="630" y="85" text-anchor="middle">top-k (exact)</text>
<path class="dg-arrow" d="M490,80 H546" marker-end="url(#flat-a)"/>
</svg>
<figcaption>Every live vector is read and scored; deleted ids are skipped.</figcaption>
</figure>

- **Storage.** Each vector is one fixed-size record of `4 × D` bytes
  (little-endian f32) in `vectors.bin`, after a 24-byte versioned header.
  A `.crc` sidecar holds one CRC32C per 1 MiB block (ADR 0007), so damage
  is detected on open instead of returned as results.
- **Scan.** The file is memory-mapped and scored in blocks of 256 vectors
  in parallel. Each block keeps its own best `k`, and the blocks are merged.
- **Scores.** Results are true cosine similarities in `[-1, 1]`.
- **Durability.** After `persist()` returns, every vector added before it
  survives a crash; see [Durability and persist()](/srvdb/guides/durability/).

## Try it in SrvDB

```python
import random
import srvdb

rng = random.Random(7)
vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(1000)]
ids = [f"doc-{i}" for i in range(len(vectors))]

db = srvdb.open("./flat-db", dimension=128, mode="flat")
db.add(ids, vectors, ["{}"] * len(ids))
db.persist()

top = db.search(vectors[42], k=3)
print(top[0][0], round(top[0][1], 3))
# doc-42 1.0
```

## Trade-offs

| | Flat |
|---|---|
| Recall | 1.0 by definition: the ranking is exact |
| Search work | Reads all `n × 4 × D` bytes for every query; grows linearly with `n` |
| Bytes per vector | `4 × D`: 512 at D = 128, 1536 at D = 384, 3072 at D = 768 |
| Build or training | None |

## When to choose it

- Collections where a linear scan is fast enough for you: measure your
  own query latency first, because it is often the case for local data.
- Whenever results must be exact, or as ground truth when you measure the
  recall of [HNSW](/srvdb/concepts/hnsw/), [SQ8](/srvdb/concepts/sq8/),
  [PQ](/srvdb/concepts/pq/) or [RaBitQ](/srvdb/concepts/rabitq/).
- Move on when the scan is too slow (try RaBitQ or HNSW) or the files are
  too large (try SQ8 or PQ). [Choosing an index mode](/srvdb/guides/index-modes/)
  shows a recall check.

## FAQ

### Is flat search in SrvDB exact?

Yes. Every live vector is scored by cosine similarity against the query, so
the top k results are the true nearest neighbours.

### Does flat mode need an index build or training?

No. Vectors are searchable as soon as `add()` returns; there is nothing to
build, train or tune.

### How much disk does flat mode use per vector?

4 bytes per dimension in `vectors.bin`, for example 3072 bytes at dimension
768, plus metadata and a small checksum sidecar.

## Related concepts

:::note[Related concepts]
- [RaBitQ](/srvdb/concepts/rabitq/): a faster scan that keeps flat's exact scores through rerank.
- [HNSW](/srvdb/concepts/hnsw/): search without reading every vector.
- [Compare index modes](/srvdb/concepts/comparison/).
- Reference: [`srvdb.open`](/srvdb/reference/python/open/), [`Database.search`](/srvdb/reference/python/database-search/), [on-disk files](/srvdb/reference/files/).
:::

**Next:** [What is HNSW?](/srvdb/concepts/hnsw/)

## Papers and further reading

Exact search needs no paper; every approximate method on these pages is
judged against it. The concept pages for each approximate mode list the
papers behind them.
