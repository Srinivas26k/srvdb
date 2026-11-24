# 12 Edge and small memory

For a device with little RAM and tight time limits. The same dataset
(`make_blobs`, 128 dimensions; 50,000 vectors, or 4,000 with `--fast`) is
stored as flat float32, SQ8 and RaBitQ at 4, 2 and 1 bits. For each the
script reports estimated resident memory (`db.memory_report()`), bytes on
disk, recall@10 against exact search and latency.

On the 1-bit database it also runs:

- `db.search(q, k, certify=True)`: rescore only the candidates the RaBitQ
  error bounds cannot settle;
- `db.search(q, k, deadline_ms=0.05)`: stop scanning codes after that
  time; `outcome.completeness` says how much was scanned.

Finally it copies a database without its `.crc` checksum files, as a
careless deployment script might, shows that `srvdb.open` refuses it, and
repairs it with `srvdb.rebuild_checksums(path)`.

```bash
python examples/gallery/12_edge/edge.py [--fast]
```

Output: one line per encoding, the certify and deadline results, the
repair steps, and `output/12_edge.png`: recall@10 against estimated memory
(log scale), one labelled point per encoding.

RaBitQ keeps the full vectors on disk for reranking, so it saves RAM, not
disk. See the [RaBitQ guide](https://srinivas26k.github.io/srvdb/guides/rabitq/).
