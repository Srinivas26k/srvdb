---
title: Filtered search
description: Search only the vectors whose metadata matches a condition.
---

Pass `where=` to `db.search` or `db.search_batch` to search only the
vectors whose metadata matches. SrvDB applies the filter during the
search, so you get `k` results whenever at least `k` vectors match, even
when very few do.

```python
import json
import srvdb

db = srvdb.create("./notes", dimension=128, index="flat")
docs = [
    ("a", "jazz", 1959),
    ("b", "jazz", 1972),
    ("c", "rock", 1969),
    ("d", "rock", 1991),
]
vectors = [[float(i == j) for j in range(128)] for i in range(len(docs))]
metas = [json.dumps({"genre": g, "year": y}) for _, g, y in docs]
db.add([d[0] for d in docs], vectors, metas)

query = vectors[0]
print([id_ for id_, _ in db.search(query, k=10, where={"genre": "jazz"})])
# ['a', 'b']
print(sorted(id_ for id_, _ in db.search(query, k=10, where={"year": {"$gte": 1970}})))
# ['b', 'd']
```

## Filter syntax

A filter is a dict. Each key is a field name and each value is either
the value to match or a dict of operators. Several keys must all match.

| Filter | Matches when |
| --- | --- |
| `{"genre": "jazz"}` | `genre` equals `"jazz"` (strings, numbers, bools, `None`) |
| `{"year": {"$gt": 1960}}` | `$gt`, `$gte`, `$lt`, `$lte`: numbers with numbers, strings with strings |
| `{"year": {"$gte": 1960, "$lt": 1970}}` | every operator on the field holds |
| `{"genre": {"$ne": "rock"}}` | not equal, or the field is missing |
| `{"genre": {"$in": ["jazz", "blues"]}}` | equal to one of the values |
| `{"genre": {"$nin": ["rock"]}}` | none of the values, or the field is missing |
| `{"$or": [{...}, {...}]}` | any of the filters matches |
| `{"$and": [{...}, {...}]}` | all of the filters match |
| `{"$not": {...}}` | the filter does not match |
| `{"source.lang": "en"}` | `lang` inside the object stored under `source` |

Numbers compare by value, so `1` matches `1.0`; integers compare
exactly, so `2**53 + 1` does not match `2**53`. A bool is not a number,
so `True` does not match `1`. Dict keys must be `str`. A filter holds at
most 1,000 conditions, and each `$in` or `$nin` list at most 10,000
values.

```python
import json
import srvdb

db = srvdb.create("./books", dimension=128, index="flat")
rows = [
    {"source": {"lang": "en"}, "pages": 120},
    {"source": {"lang": "de"}, "pages": 300},
    {"source": {"lang": "en"}, "pages": 640},
]
vectors = [[float(i == j) for j in range(128)] for i in range(len(rows))]
db.add(["x", "y", "z"], vectors, [json.dumps(r) for r in rows])

where = {"$or": [{"source.lang": "de"}, {"pages": {"$gt": 500}}]}
print(sorted(id_ for id_, _ in db.search(vectors[0], k=5, where=where)))
# ['y', 'z']
```

## Errors

An invalid filter raises `ValueError` naming the problem, before any
search runs: an empty dict, an unknown operator such as `$regex`, a list
given directly as a value (use `$in`), a bool given to `$gt`, `NaN`, or a
value JSON cannot encode.

```python
import srvdb

db = srvdb.create("./errs", dimension=128, index="flat")
try:
    db.search([1.0] * 128, k=3, where={"genre": {"$like": "ja%"}})
except ValueError as err:
    print(err)
# invalid input: unknown operator '$like' on filter field 'genre'; use one of $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
```

## Performance

- Each filtered call reads every metadata record once to find the
  matches (O(n)), then searches only those. `db.search_batch(queries, k,
  where=...)` reads the metadata once for the whole batch.
- With an HNSW index the graph walk skips vectors that do not match. When
  few enough match that scoring them directly is cheaper, SrvDB scores
  them exactly, so selective filters return exact results. See
  [ADR 0010](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0010-filtered-search.md).
- Other index modes score the matching vectors exactly.

## In LangChain and LlamaIndex

The [framework adapters](/srvdb/guides/frameworks/) pass their filters
to SrvDB: LangChain dict filters (including the operators above) and
LlamaIndex `MetadataFilters`, `node_ids` and `doc_ids`. A LangChain filter
given as a Python callable still runs on the search results.

## See also

- [`db.search`](/srvdb/reference/python/db-search/)
- [`db.search_batch`](/srvdb/reference/python/db-search-batch/)
