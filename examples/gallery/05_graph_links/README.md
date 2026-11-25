# 05 Graph links

Chunked documents often split a subject from its details: the heading
chunk names "the primary pump unit0", the next chunk holds the steps and
never repeats the name. A vector search for the subject finds the heading
but not the steps.

The script stores synthetic maintenance manuals as heading/body chunk
pairs, links them with `db.link(heading_id, "next", body_id)`, and for
each heading asks "how do I service the ...". It counts how often the
body chunk is returned by

- `db.search(vector, k=3)`, and
- `db.search(vector, k=3, expand=("next", 1))`, which appends the records
  linked from the hits and returns `(id, score, hop)` tuples.

```bash
python examples/gallery/05_graph_links/graph_links.py [--fast]
```

Output: `db.neighbors("h0")`, an example expanded result with hops, the
two hit rates, and `output/05_graph_links.png` with one bar for each.

Links are stored with the database and saved by `save()`. `expand=`
cannot be combined with `where=`. See the
[links guide](https://srinivas26k.github.io/srvdb/guides/links/).
