# 03 Filtered search

Each record gets a random `bucket` from 0 to 99, so
`where={"bucket": {"$lt": s}}` keeps about `s` percent of the records.
For `s` in 1, 2, 5, 10, 25, 50 and 100 the script searches an HNSW
database with the filter and compares the top 10 with exact search over
only the matching records.

```bash
python examples/gallery/03_filtered_search/filtered_search.py [--fast]
```

Output: one line per selectivity (share matching, recall@10, mean latency)
and `output/03_filtered_search.png`: recall@10 (solid) and mean latency
(dashed, right axis) against the share of matching records on a log scale.

The point it measures: a filter applied after an approximate search can
return fewer than `k` results or miss matches when few records qualify.
SrvDB evaluates `where=` inside the search, so recall should stay at the
level of the unfiltered search however selective the filter is; the chart
shows whether it does on this run. Operators: `$eq $ne $gt $gte $lt $lte
$in $nin $and $or $not`, and `"a.b"` for nested fields. See the
[filtered search guide](https://srinivas26k.github.io/srvdb/guides/filtered-search/).
