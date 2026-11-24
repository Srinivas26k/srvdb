---
title: Quickstart
description: Create, search, update and save vectors in five minutes.
---

This walkthrough uses random vectors so it runs anywhere. In a real app the
vectors come from an embedding model; see
[Build a local RAG app](/srvdb/guides/local-rag/).

## 1. Create a database and add vectors

```python
import json
import random
import srvdb

DIM = 128

def vec():
    return [random.random() - 0.5 for _ in range(DIM)]

docs = {f"doc{i}": vec() for i in range(1000)}

with srvdb.create("./quickstart-db", dimension=DIM) as db:
    db.add(
        ids=list(docs),
        embeddings=list(docs.values()),
        metadatas=[json.dumps({"n": i}) for i in range(1000)],
    )
```

`srvdb.create` makes a new database; the dimension must match your
embedding model. The index defaults to HNSW; pass `index="flat"`,
`"sq8"`, `"rabitq"` and so on to choose another
([Choosing an index mode](/srvdb/guides/index-modes/)). Every id must be new
and unique; metadata must be a JSON object string.

The `with` block **saves** the database when it ends and releases its lock.
If the block ends with an exception, nothing since the last save is made
durable, so a half-finished batch never reaches disk.

## 2. Open it and search

```python
with srvdb.open("./quickstart-db") as db:
    for doc_id, score in db.search(docs["doc42"], k=3):
        print(doc_id, round(score, 3), db.get(doc_id))
# doc42 1.0 {"n": 42, ...}
```

`srvdb.open` takes only the path: the dimension and index were recorded by
`create`. Results are `(id, score)` pairs, best first. The score is the
cosine similarity: 1.0 means the same direction, 0 unrelated, -1 opposite.

## 3. Save without a `with` block

```python
db = srvdb.open("./quickstart-db")
db.upsert(ids=["doc42"], embeddings=[vec()], metadatas=['{"n": 42, "v": 2}'])
db.delete(["doc7"])
db.save()
print(db.count())   # 999
db.close()
```

Searches see new vectors immediately, but **only `save()` makes them
crash-safe**. Call it after a batch of writes, not after every single add.
`close()` releases the lock without saving.

## 4. Reopen

```python
with srvdb.open("./quickstart-db") as db:
    print(db.count())   # 999: exactly what was saved
```

## 5. Delete the database

```python
srvdb.delete("./quickstart-db")
```

This removes the directory, but only if it holds a closed SrvDB database
and nothing else.

## Next

- [Choose an index mode](/srvdb/guides/index-modes/) for larger data.
- [Concepts](/srvdb/concepts/comparison/): how each index mode works inside SrvDB.
- [Durability](/srvdb/guides/durability/): exactly what `save()` guarantees.
- [Python API](/srvdb/reference/python/): every verb and method.
