---
title: Delete, upsert and compact
description: Remove and replace vectors, and reclaim their space.
---

## Delete

```python
removed = db.delete(["doc3", "doc9", "not-there"])   # -> 2
```

Deleted vectors disappear from every search immediately, and searches still
return `k` results. Unknown ids are ignored. The delete becomes durable at the
next `persist()`; without it, a crash brings the vectors back.

## Upsert

```python
db.upsert(ids=["doc3"], embeddings=[new_vector], metadatas=['{"v": 2}'])
```

Existing ids point to the new vector (the old one is deleted); new ids are
added. Use `add` when ids must be new: it rejects existing ids.

## Compact

Deleted vectors still occupy disk space until you compact:

```python
removed = db.compact()   # number of vectors physically removed
```

`compact()` persists first, rewrites the files without deleted vectors and
rebuilds the HNSW or IVF index, so it takes time proportional to the data.
Run it occasionally, for example after deleting a large share of the
collection. Your string ids do not change.

It is crash-safe: a compaction interrupted by a crash is finished or discarded
on the next open
([ADR 0004](https://github.com/Srinivas26k/srvdb/blob/master/docs/adr/0004-delete-update-compact.md)).
