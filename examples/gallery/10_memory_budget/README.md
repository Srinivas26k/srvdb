# 10 Memory budget

`srvdb.create(path, dimension=128, index="auto", memory_mb=8)` (1 MiB with
`--fast`) starts flat and exact. When an add would take the estimate past
80 % of the budget, it re-encodes every vector to RaBitQ at 4, then 2,
then 1 bit per dimension; past that, `add` raises `ValueError` and writes
nothing.

The script adds batches until that happens. After every batch it records
`db.memory_report()` (mode, bits, `total_bytes`) and recall@10 against
exact search, then reopens the database to show the mode and budget were
recorded.

```bash
python examples/gallery/10_memory_budget/memory_budget.py [--fast]
```

Output: a line per mode change (and every fifth batch), the refusal
message, and `output/10_memory_budget.png`: estimated memory with the 80 %
limit (top) and recall@10 (bottom) against records stored, with dotted
lines where the mode changed.

The memory figures are estimates from `memory_report()`, not process RSS.
See the [memory budget guide](https://srinivas26k.github.io/srvdb/guides/memory-budget/).
