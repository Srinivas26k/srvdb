---
title: Links between records
description: Connect records with typed, directed links and follow them during search.
---

Records often relate to each other: a paper cites another, a chunk belongs
to a section, a ticket duplicates a ticket. `db.link(a, "cites", b)` stores
that relationship, `db.neighbors(id)` reads it back, and
`db.search(q, k, expand=("cites", 2))` follows it from the search hits.

```python
import srvdb

db = srvdb.create("./papers", dimension=128, index="flat")
vectors = [[float(i == j) for j in range(128)] for i in range(4)]
db.add(["intro", "method", "results", "appendix"], vectors, ["{}"] * 4)

db.link("intro", "cites", "method")
db.link("method", "cites", "results")
db.save()  # links are durable from here, together with the records

print(db.neighbors("intro"))
# [('cites', 'method')]
print([(id_, hop) for id_, _, hop in db.search(vectors[0], k=1, expand=("cites", 2))])
# [('intro', 0), ('method', 1), ('results', 2)]
```

## Links

- A link is directed: `a -cites-> b` says nothing about `b -cites-> a`.
- The link type is a name of lowercase letters, digits and `_`, starting
  with a letter, at most 64 characters (`cites`, `part_of`, `v2`). Anything
  else raises `ValueError`.
- Linking twice returns `False` and changes nothing; `db.unlink` returns
  whether the link existed.
- A record may have at most 1,024 outgoing links and a database at most
  10,000,000 links. Incoming links are not limited per record.
- Both ends must exist, and a record cannot link to itself.

## Durability

Links follow the same rule as records: they are visible at once and
durable after `db.save()`, written in the same commit as the records. After
a crash the database reopens with exactly the links of the last `save()`,
never a link to a record that was not saved. Links made after the last
`save()` are gone after reopening, however the process ended.

## Delete, upsert and compact

```python
import srvdb

db = srvdb.create("./graph", dimension=128, index="flat")
vectors = [[float(i == j) for j in range(128)] for i in range(3)]
db.add(["a", "b", "c"], vectors, ["{}"] * 3)
db.link("a", "cites", "b")
db.link("b", "cites", "c")

db.upsert(["b"], [vectors[1]], ['{"v": 2}'])
print(db.neighbors("a"), db.neighbors("b"))
# [('cites', 'b')] [('cites', 'c')]

db.delete(["b"])
db.compact()
print(db.neighbors("a"))
# []
```

- Deleting a record removes every link it is part of, in both directions.
- `upsert` keeps the record's links on its new version.
- `compact` keeps links intact while it renumbers records internally.

## Search expansion

`expand=(link_type, hops)` runs the normal search for the `k` hits, then
follows outgoing links of `link_type` breadth-first:

- `hops` is 1 to 3. Pass `(link_type, hops, max_results)` to cap the
  number of linked records added (default 100, at most 1,000).
- Results become `(id, score, hop)` tuples: hits have hop 0, linked
  records follow, ordered by hop. Each record appears once, at its
  smallest hop.
- A linked record's score is the score of the record it was reached from,
  not its own similarity to the query.
- `expand` cannot be combined with `where`.

The design and its guarantees are recorded in ADR 0016.
