---
title: db.enable_keyword_search
description: Index a metadata field for keyword (BM25) search.
---

Index a metadata field for keyword (BM25) search with `db.search(text=...)`.

```python
db.enable_keyword_search(field: str = "text") -> None
```

## Parameters

- **field** : *str, default `"text"`*
  Name of a string field in each record's metadata JSON. `"a.b"` names a nested field; a list of strings is indexed as one text.

## Raises

- **ValueError**
  `field` is empty or has an empty part (`"a..b"`), or is longer than 256 bytes.

- **OSError**
  Reading the metadata failed.

## Notes

- Reuses the saved index when it covers the same field; otherwise builds it from metadata, reading every record once.
- The index is written by the next [`save()`](/srvdb/reference/python/db-save/) and used again after `srvdb.open()` without calling this again.
- Records without a string value in the field are not indexed.
- Call it again to rebuild a damaged index file.

## Examples

```python
import srvdb

db = srvdb.create("./docs", dimension=128)
v = [1.0] + [0.0] * 127
db.add(["a"], [v], ['{"text": "error E0308"}'])
db.enable_keyword_search("text")
print(db.search(None, k=1, text="e0308")[0][0])
# a
```

## See also

- [Keyword and hybrid search](/srvdb/guides/keyword-search/)
- [`db.search`](/srvdb/reference/python/db-search/)
