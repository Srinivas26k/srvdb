# SrvDB: Embedded Vector Database for Offline AI Applications

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.99+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/index.md)

**[📚 Read the Full Documentation](docs/index.md)**

SrvDB is an embedded vector database written in Rust, with Python bindings.
It runs inside your process: no server, no network access, no background
service. It is built for local and offline AI applications such as RAG over
private documents, desktop tools and air-gapped systems.

Its first duty is to keep committed data safe; speed and features come after
that.

---

## What you get

- **Crash-safe storage.** After `persist()` returns, every earlier write
  survives a crash or power loss. After a crash the database reopens in
  exactly the state of the last `persist()`: later writes are absent, never
  half-present. See [ADR 0003](docs/adr/0003-durability.md).
- **Delete, upsert and compact.** Deletes take effect in search immediately
  and become durable at `persist()`; `compact()` reclaims the space.
- **Several index modes.** Exact flat search, HNSW, IVF, and SQ8 or PQ
  compression, all searched by cosine similarity. Scores are true cosine
  similarities in [-1, 1].
- **Validated input.** Wrong dimensions, NaN or infinite values, `k < 1`,
  duplicate ids and non-JSON metadata are rejected with clear errors.
- **Safe to share between threads.** Every Python method releases the GIL;
  searches run in parallel. A database can be open in only one place at a
  time; a second open raises `BlockingIOError`.
- **Indexes are derived data.** HNSW and IVF indexes can always be rebuilt
  from the stored vectors. A missing or damaged HNSW graph is rebuilt on
  open; a damaged IVF index is reported as a `Corrupt` error.

---

## Installation

```bash
pip install srvdb
```

Wheels are built for Python 3.9 to 3.14 on Linux (x86_64, aarch64), macOS
(x86_64, arm64) and Windows (x64).

### Build from Source

Installing the wheel from PyPI needs no Rust toolchain. Building from
source (an unsupported platform, the sdist, or development) needs
**Rust 1.99 or newer** (`rustup update stable`) and
[maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/Srinivas26k/srvdb
cd srvdb
cargo test --workspace        # Rust core
pip install maturin
maturin develop --release     # builds and installs the Python package
```

---

## Quick Start

```python
import json
import srvdb

db = srvdb.Database("./vectors", dimension=384, mode="flat")

db.add(
    ids=["doc1", "doc2"],
    embeddings=[embedding_1, embedding_2],     # lists of 384 floats
    metadatas=[json.dumps({"source": "a.txt"}), json.dumps({"source": "b.txt"})],
)
db.persist()                                   # durable from here on

for doc_id, score in db.search(query_embedding, k=5):
    print(doc_id, score, db.get(doc_id))
```

Any embedding model works as long as its dimension is between **128 and
4096**; use the model's own dimension, with no padding.

---

## Index modes

| Mode | Search | Stored per vector | Notes |
|------|--------|-------------------|-------|
| `flat` (default) | Exact scan | 4 bytes × dimension | Exact results. |
| `hnsw` | Approximate graph | Full vectors + graph | Tune with `set_ef_search()`. |
| `ivf` | Partitions, then graph | Full vectors + index | Call `train_ivf()` once data is loaded. |
| `sq8` | Scan of 8-bit codes | 1 byte × dimension | 4× smaller; approximate scores. |
| `pq` | Scan of PQ codes | dimension ÷ 8 bytes (÷ 4 if not divisible by 8) | Up to 32× smaller; trains on the first batch (at least 256 vectors). |
| `auto` | Flat, then HNSW | Full vectors | Checks size at each `persist()` and builds an HNSW graph for large databases; never compresses. |

Compressed and approximate modes trade recall for size or speed. **How much
depends on your data**: product quantization in particular loses accuracy
on tightly clustered embeddings. Measure recall on a sample of your own
data against `flat` before choosing one. From Rust, HNSW can also be built
over PQ storage (`SrvDB::new_with_hnsw_quantized`).

More detail: [Indexing modes](docs/concepts/modes.md).

---

## API overview

```python
# Constructors
db = srvdb.Database(path, dimension=1536, mode="flat")   # see modes above
db = srvdb.Database.new_hnsw(path, dimension=1536, m=16, ef_construction=200, ef_search=50)
db = srvdb.Database.new_scalar_quantized(path, dimension, training_vectors)
db = srvdb.Database.new_product_quantized(path, dimension, training_vectors)

# Writes (visible to search immediately, durable after persist())
db.add(ids, embeddings, metadatas) -> int      # new ids only
db.upsert(ids, embeddings, metadatas) -> int   # insert or replace
db.delete(ids) -> int                          # number actually deleted
db.persist()                                   # crash-safe commit
db.compact() -> int                            # reclaim space from deletes

# Reads
db.search(query, k) -> list[tuple[str, float]]          # best first
db.search_batch(queries, k) -> list[list[tuple[str, float]]]
db.get(id) -> str | None                                 # metadata JSON
db.count() -> int
db.get_all_ids() -> list[str]

# Tuning
db.set_mode("hnsw")                    # 'flat', 'hnsw', 'sq8', 'ivf', 'auto'
db.set_ef_search(100)                  # HNSW: higher is more accurate, slower
db.configure_ivf(nlist=100, nprobe=10)
db.train_ivf()
```

Errors: bad input raises `ValueError`, I/O failures raise `OSError`, a
database already open elsewhere raises `BlockingIOError`, and internal
faults raise `RuntimeError`. Full reference: [Python API](docs/api/python.md).

---

## Performance

SrvDB 0.3 does not publish performance numbers yet. Earlier README tables
were measured on versions where some modes did not work as labelled (for
example, the "HNSW" rows measured a flat scan, and HNSW with PQ returned no
results), so they have been removed.

New numbers will be published only from a committed, reproducible benchmark,
with the hardware, OS, dataset, commit and command listed next to every
figure. Until then, measure on your own hardware:

```bash
pip install "srvdb[bench]"
python benchmarks/universal_benchmark.py
```

See [Benchmarking](docs/guides/benchmarking.md).

---

## Known limitations

- **One open instance per database.** A database directory can be open in
  one `Database` (or Rust `SrvDB`) at a time, across all processes.
- **`persist()` is the commit point.** Writes after the last `persist()`
  are lost on a crash, by design.
- **Approximate modes need checking.** Recall of `hnsw`, `ivf`, `sq8` and
  `pq` varies with data and parameters; measure it on your data.
- **Single node.** No replication or sharding; SrvDB is a library inside
  your application.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines,
and [AGENTS.md](AGENTS.md) for the engineering rules every change follows.
Benchmark results from different hardware are welcome in
[GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions).

---

## License

SrvDB is licensed under the [Apache License 2.0](LICENSE): free to use,
modify and ship in open-source and commercial products, with an explicit
patent grant. Releases up to and including 0.2.0 were published under
AGPL-3.0.

Commercial support and consulting: **srinivasvarma764@gmail.com**

---

## Acknowledgments

SrvDB relies on the following open-source projects:

- [SimSIMD](https://github.com/ashvardanian/simsimd) - SIMD distance kernels
- [Rayon](https://github.com/rayon-rs/rayon) - Data parallelism
- [PyO3](https://github.com/PyO3/pyo3) - Python-Rust bindings
- [redb](https://github.com/cberner/redb) - Embedded key-value store
- [parking_lot](https://github.com/Amanieu/parking_lot) - Fast synchronization primitives

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Srinivas26k/srvdb/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions)
- **Email:** srinivasvarma764@gmail.com

For bug reports or feature requests, please include system information and minimal reproduction code.
