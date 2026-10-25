---
title: Quickstart
description: Store, search, update and persist vectors in five minutes.
---

This walkthrough uses random vectors so it runs anywhere. In a real app the
vectors come from an embedding model; see
[Build a local RAG app](/srvdb/guides/local-rag/).

## 1. Open a database

```python
import json
import random
import srvdb

DIM = 128
db = srvdb.Database("./quickstart-db", dimension=DIM)
```

The directory is created if needed. Opening an existing database loads it.
The dimension must be between 128 and 4096 and match your embedding model.

## 2. Add vectors

```python
def vec():
    return [random.random() - 0.5 for _ in range(DIM)]

docs = {f"doc{i}": vec() for i in range(1000)}
db.add(
    ids=list(docs),
    embeddings=list(docs.values()),
    metadatas=[json.dumps({"n": i}) for i in range(1000)],
)
```

Every id must be new and unique; metadata must be a JSON object string.

## 3. Search

```python
for doc_id, score in db.search(docs["doc42"], k=3):
    print(doc_id, round(score, 3), db.get(doc_id))
# doc42 1.0 {"n": 42, ...}
```

Results are `(id, score)` pairs, best first. The score is the cosine
similarity: 1.0 means the same direction, 0 unrelated, -1 opposite.

## 4. Make it durable

```python
db.persist()
```

Searches see new vectors immediately, but **only `persist()` makes them
crash-safe**. Call it after a batch of writes, not after every single add.

## 5. Update and delete

```python
db.upsert(ids=["doc42"], embeddings=[vec()], metadatas=['{"n": 42, "v": 2}'])
db.delete(["doc7"])
db.persist()
print(db.count())   # 999
```

## 6. Reopen

```python
del db
db = srvdb.Database("./quickstart-db", dimension=DIM)
print(db.count())   # 999: exactly what was persisted
```

## Next

- [Choose an index mode](/srvdb/guides/index-modes/) for larger data.
- [Durability](/srvdb/guides/durability/): exactly what `persist()` guarantees.
