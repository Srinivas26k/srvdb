# 01 Getting started

The smallest complete SrvDB program, on the scikit-learn digits images
(1797 images of 8x8 pixels, so 64-dimensional vectors).

1. `srvdb.create(path, dimension=64, index="flat")` in a `with` block,
   which saves and closes the database on exit.
2. `db.add(ids, vectors, metadatas)` with the digit label as JSON metadata.
3. `srvdb.open(path)` reopens with the recorded settings; `db.search`
   returns `(id, cosine)` pairs; `db.get(id)` returns the metadata.
4. `db.search(..., where={"label": 8})` filters inside the database.
5. `srvdb.delete(path)` removes the closed database directory.

```bash
python examples/gallery/01_getting_started/getting_started.py [--fast]
```

Output: the five nearest images to a held-out image with their labels and
scores, the closest images with another label, and the label a 5-nearest
neighbour vote assigns. No chart.

Next: [02 index modes](../02_index_modes/) to choose an index.
