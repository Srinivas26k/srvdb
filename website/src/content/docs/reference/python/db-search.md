---
title: db.search
description: Find the k nearest vectors to a query.
---

Find the k nearest vectors to a query.

```python
db.search(query: Sequence[float], k: int, *, where: dict | None = None,
          certify: bool = False, deadline_ms: float | None = None)
    -> list[tuple[str, float]] | SearchOutcome
```

## Parameters

- **query** : *sequence of float*
  Query vector with the database dimension and only finite values (a list or 1-D NumPy array).

- **k** : *int*
  Maximum number of results, at least 1.

- **where** : *dict, optional, keyword-only*
  Metadata filter, for example `{"genre": "jazz"}` or `{"year": {"$gte": 1950}}`. Operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`, `$not`; `"a.b"` names a nested field. See [Filtered search](/srvdb/guides/filtered-search/). Default `None` searches every vector.

- **certify** : *bool, keyword-only, RaBitQ modes only*
  Use the RaBitQ error bounds: certify results whose interval clears the k-th lower bound and rescore only the uncertain candidates on the full vectors. See [RaBitQ, step 5](/srvdb/concepts/rabitq/#5-certified-and-deadline-search-optional). Default `False`.

- **deadline_ms** : *float, optional, keyword-only, RaBitQ modes only*
  Stop after this many milliseconds and return the best results found. The clock is checked between 32-vector scan blocks and between 256-vector rescoring batches. Default `None`, no deadline.

## Returns

`list[tuple[str, float]]`: up to `k` `(id, score)` pairs, best first. Scores are cosine similarities computed in float32, so an identical vector scores about 0.99999994 rather than exactly 1.0. With `where`, only matching vectors are returned, and `k` of them whenever at least `k` match. Fewer than `k` pairs come back when fewer vectors (or matches) exist; an empty database returns `[]`.

With `certify=True` or `deadline_ms`, a `SearchOutcome`. It iterates, indexes (integers and slices) and has a `len()` like the list above (`out.results` returns that plain list), and also has these attributes:

- `completeness` (float): fraction of live vectors scanned, 1.0 unless the deadline passed.
- `confidence` (float or None): with `certify=True`, 0.99, a nominal and approximate lower bound on the probability, over a hypothetical random rotation seed, that the results are exactly the top k of the scanned vectors. It is not a per-query guarantee, not a long-run rate for this database's fixed seed, and not joint across queries. `None` without `certify`, and when the rerank cap (`max(k × rerank, 1000)` candidates) or the deadline cut the rescoring short; `certified` is then 0.
- `certified`, `reranked`, `scanned` (int): results certified without rescoring, candidates rescored, and vectors scanned.

## Raises

- **ValueError**
  `k` is 0, or a query does not match the database dimension or contains NaN or infinity, or `where` is not a valid filter (empty, an unknown operator, a wrong operand type, a value JSON cannot encode). Also raised for a negative `k`; for `certify` or `deadline_ms` on a database that is not in a RaBitQ mode or combined with `where`; and for a negative or non-finite `deadline_ms`.

- **RuntimeError**
  An internal fault occurred.

## Notes

- Sees every write whose call has returned, persisted or not.
- Results are exact in `"flat"` mode and approximate in the other modes. In `"pq"` mode scores are approximate too.
- A `where` filter reads every metadata record once per call (O(n)); `search_batch` shares that pass across queries.
- Thread safety: releases the GIL and takes a shared lock, so calls from several threads run in parallel; they wait while a write is running. See [Threads and processes](/srvdb/guides/concurrency/).

## Examples

```python
import srvdb

db = srvdb.create("./vectors", dimension=128)
v = [1.0] + [0.0] * 127   # tiny 128-dimension vectors
w = [0.0, 1.0] + [0.0] * 126
db.add(["a", "b"], [v, w], ["{}", "{}"])
print(db.search(v, k=2))
# [('a', 0.9999999403953552), ('b', 0.0)]

db.add(["c"], [w], ['{"kind": "note"}'])
print(db.search(v, k=2, where={"kind": "note"}))
# [('c', 0.0)]
```

With a RaBitQ database:

```python
import random
import srvdb

rng = random.Random(3)
vecs = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(1000)]
db = srvdb.create("./rq", dimension=128, index="rabitq")
db.add([f"d{i}" for i in range(1000)], vecs, ["{}"] * 1000)
out = db.search(vecs[5], k=1, certify=True, deadline_ms=10_000)
print(out[0][0], out.completeness, out.confidence)
# d5 1.0 0.99
```

## See also

- [Filtered search](/srvdb/guides/filtered-search/): Filter syntax and performance.
- [`db.search_batch`](/srvdb/reference/python/db-search-batch/): Run several searches in one call.
- [`db.get`](/srvdb/reference/python/db-get/): Return the metadata stored for an id.
- [`db.set_ef_search`](/srvdb/reference/python/db-set-ef-search/): Set HNSW's search candidate list size.
