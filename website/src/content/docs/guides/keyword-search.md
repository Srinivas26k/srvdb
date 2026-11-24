---
title: Keyword and hybrid search
description: Match exact words with BM25 and combine them with vector search.
---

Embeddings blur exact words: product codes, names and error messages.
SrvDB can also rank records by the words in one metadata field (BM25), and
fuse that ranking with vector search. It needs no model, no network and
no second database.

```python
import json
import srvdb

db = srvdb.create("./notes", dimension=128, index="flat")
texts = {
    "a": "Install fails with error E0308 on Windows",
    "b": "How to install on macOS",
    "c": "Release notes for version 2",
}
vectors = [[float(i == j) for j in range(128)] for i in range(len(texts))]
metas = [json.dumps({"text": t}) for t in texts.values()]
db.add(list(texts), vectors, metas)

db.enable_keyword_search("text")      # index the "text" field
print([i for i, _ in db.search(None, k=3, text="error E0308")])
# ['a']
print([i for i, _ in db.search(vectors[1], k=2, text="install")])
# ['b', 'a']
db.save()                             # also saves the keyword index
```

## Three kinds of search

| Call | Ranking | Score |
| --- | --- | --- |
| `db.search(query, k)` | Vector similarity | Cosine similarity |
| `db.search(None, k, text="...")` | BM25 over the indexed field | BM25 score |
| `db.search(query, k, text="...")` | Both, fused with Reciprocal Rank Fusion | Fused score |

`where=` works with all three and restricts both rankings. Hybrid search
takes the best `max(2k, 50)` results of each ranking and scores every
record `1 / (60 + rank)` per list it appears in, so a record near the top
of both lists wins. The fused score only orders results; it is not a
similarity.

## What gets indexed

`db.enable_keyword_search(field="text")` indexes one field of the metadata
JSON. `"a.b"` names a nested field and a list of strings is indexed as one
text. Records without a string value in that field are not indexed.
Call it once; the index is saved by `save()` and used again after
`srvdb.open()`. Calling it with another field rebuilds the index.

Words are runs of letters and digits in any script, lowercased. There is
no stemming and there are no stop words: `install` does not match
`installed`. Languages written without spaces (Chinese, Japanese, Thai)
only match whole runs of text. A query may hold up to 64 KiB of text and
1,024 distinct words; longer ones raise `ValueError`.

## Durability

The keyword index is derived from metadata and adds no guarantee of its
own: after a crash it reflects exactly the records of the last `save()`.
If the index file is damaged, keyword searches raise `RuntimeError`; call
`enable_keyword_search()` again to rebuild it from metadata. See
[ADR 0015](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0015-keyword-search.md)
for the file format.

## Performance

Adds and deletes are indexed on the next keyword search or `save()`.
Keyword searches on one database run one at a time; vector searches are not
blocked by them. `scripts/bench_keyword.py` compares speed and quality with
SQLite FTS5 on your machine.
