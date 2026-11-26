<p align="center">
  <a href="https://srinivas26k.github.io/srvdb/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/brand/srvdb-logo-horizontal-dark.svg">
      <img src="assets/brand/srvdb-logo-horizontal.svg" alt="SrvDB" height="110">
    </picture>
  </a>
</p>

<p align="center"><b>An embedded, crash-safe vector database for local and offline AI. Rust core, Python first.</b></p>

<p align="center">
  <a href="https://pypi.org/project/srvdb/"><img src="https://img.shields.io/pypi/v/srvdb?color=F2B134&labelColor=1B2A4A" alt="PyPI"></a>
  <a href="https://pypi.org/project/srvdb/"><img src="https://img.shields.io/pypi/pyversions/srvdb?color=F2B134&labelColor=1B2A4A" alt="Python versions"></a>
  <a href="https://github.com/Srinivas26k/srvdb/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Srinivas26k/srvdb/ci.yml?branch=master&label=CI&labelColor=1B2A4A" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-C8102E?labelColor=1B2A4A" alt="License: Apache-2.0"></a>
  <a href="https://doi.org/10.5281/zenodo.24568281"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24568281-1B2A4A?labelColor=1B2A4A&color=F2B134" alt="Paper DOI: 10.5281/zenodo.24568281"></a>
</p>

<p align="center">
  <a href="https://srinivas26k.github.io/srvdb/"><b>Documentation</b></a> ·
  <a href="https://srinivas26k.github.io/srvdb/start/quickstart/">Quickstart</a> ·
  <a href="https://srinivas26k.github.io/srvdb/examples/overview/">Examples</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://srinivas26k.github.io/srvdb/project/community/">Community</a> ·
  <a href="https://srinivas26k.github.io/srvdb/project/sponsor/">Sponsor</a>
</p>

---

SrvDB (spelled S-R-V-D-B) is a library, not a service: it stores embeddings
on local disk inside your own process, with no server, no Docker and no
network access. Use it for local RAG, semantic search and AI agent memory.
It is not related to SurrealDB.

## Quickstart

```python
import srvdb

with srvdb.create("./my-db", dimension=3, index="hnsw") as db:  # saved on exit
    db.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ['{"t": "x"}', '{"t": "y"}'])

with srvdb.open("./my-db") as db:          # settings are read from disk
    print(db.search([0.9, 0.1, 0.0], k=1))  # [('a', 0.99...)]
    db.add(["c"], [[0.0, 0.0, 1.0]], ['{"t": "z"}'])
    db.save()                               # durable from here on

srvdb.delete("./my-db")
```

Metadata is a JSON string per record. Pick another index by name with its
options, for example `srvdb.create(path, 384, index="rabitq", bits=2)`.

## Install

```bash
pip install srvdb
```

Release wheels use the stable ABI (CPython 3.9 and later) for Linux x86_64
and aarch64, macOS arm64 and x86_64, and Windows x86_64. Building from
source needs Rust 1.99 or later
([installation guide](https://srinivas26k.github.io/srvdb/start/installation/)).

## Features

| Feature | What you get | Docs |
|---------|--------------|------|
| Flat | Exact cosine search | [Flat](https://srinivas26k.github.io/srvdb/concepts/flat/) |
| HNSW | Graph index, approximate, true cosine scores (default index) | [HNSW](https://srinivas26k.github.io/srvdb/concepts/hnsw/) |
| IVF | Partitions with a graph per partition, for large collections | [IVF](https://srinivas26k.github.io/srvdb/concepts/ivf/) |
| SQ8 | 1 byte per dimension, 4x smaller than flat | [SQ8](https://srinivas26k.github.io/srvdb/concepts/sq8/) |
| PQ | Up to 32x smaller codes (see Known issues) | [PQ](https://srinivas26k.github.io/srvdb/concepts/pq/) |
| RaBitQ | 1, 2 or 4 bits per dimension with reranking, no training | [RaBitQ](https://srinivas26k.github.io/srvdb/concepts/rabitq/) |
| Filtered search | `search(q, k, where={"year": {"$gte": 2020}})`, applied during the search | [Filtered search](https://srinivas26k.github.io/srvdb/guides/filtered-search/) |
| Keyword search (BM25) | `search(None, k, text="E0308")` over one metadata field | [Keyword search](https://srinivas26k.github.io/srvdb/guides/keyword-search/) |
| Hybrid search | Vector and BM25 rankings fused with Reciprocal Rank Fusion | [Keyword search](https://srinivas26k.github.io/srvdb/guides/keyword-search/) |
| Graph links | `link`, `neighbors`, and `search(..., expand=(type, hops))` | [Links](https://srinivas26k.github.io/srvdb/guides/links/) |
| Certified and deadline search | RaBitQ error bounds (`certify=True`) and a scan time limit (`deadline_ms=`) | [RaBitQ guide](https://srinivas26k.github.io/srvdb/guides/rabitq/) |
| Memory budget | `index="auto", memory_mb=50` re-encodes to fewer bits as data grows | [Memory budget](https://srinivas26k.github.io/srvdb/guides/memory-budget/) |
| Checksums and crash safety | Checksummed files, atomic saves, crash tests; `srvdb.rebuild_checksums` | [Durability](https://srinivas26k.github.io/srvdb/guides/durability/) |

Approximate modes trade accuracy for speed or size depending on your data;
measure recall against `flat` before switching
([how to choose](https://srinivas26k.github.io/srvdb/guides/index-modes/)).

## The durability promise

> When `save()` returns, every earlier `add`, `upsert` and `delete`
> survives a process crash or power loss. After a crash the database reopens
> in the state of the last `save()`: later writes are absent, never
> half-present.

Details: [Durability](https://srinivas26k.github.io/srvdb/guides/durability/)
and [ADR 0003](docs/adr/0003-durability.md).

## Use it from Rust

```toml
[dependencies]
srvdb = { git = "https://github.com/Srinivas26k/srvdb" }
```

```rust
use srvdb::{CreateOptions, IndexChoice, SrvDB, Vector, VectorEngine};

let mut db = SrvDB::create("./rust-db", 384, &CreateOptions::new(IndexChoice::Hnsw))?;
db.add(&Vector::new(vec![0.1; 384]), r#"{"title": "first"}"#)?;
db.persist()?; // durable from here on
let hits = db.search(&Vector::new(vec![0.1; 384]), 5)?;
```

More in the [Rust guide](https://srinivas26k.github.io/srvdb/guides/rust/).

## Performance

This README quotes no numbers. Published numbers come only from the
committed benchmark harness, which records hardware, OS, dataset, commit and
command: see [docs/benchmarks.md](docs/benchmarks.md) and the
[benchmarks page](https://srinivas26k.github.io/srvdb/benchmarks/). Run it on
your own machine:

```bash
cargo run --release -p srvdb --example bench
```

## Known issues

Both are targeted at 0.4.1.

- [#94](https://github.com/Srinivas26k/srvdb/issues/94): hybrid ranking can
  bury exact keyword matches. For exact codes and names, use keyword-only
  search (`search(None, k, text=...)`) or a `where` filter.
- [#95](https://github.com/Srinivas26k/srvdb/issues/95): PQ recall is low.
  Prefer RaBitQ (`index="rabitq"`) for compressed storage.

## Documentation

- [Quickstart](https://srinivas26k.github.io/srvdb/start/quickstart/) and
  [Local RAG guide](https://srinivas26k.github.io/srvdb/guides/local-rag/)
- [Python API reference](https://srinivas26k.github.io/srvdb/reference/python/)
  and [Rust API](https://srinivas26k.github.io/srvdb/reference/rust-api/)
- [Examples gallery](https://srinivas26k.github.io/srvdb/examples/overview/)
  ([source](examples/gallery/))
- [Industries](https://srinivas26k.github.io/srvdb/industries/overview/)
  and [use cases](https://srinivas26k.github.io/srvdb/start/use-cases/)
- [Roadmap](https://srinivas26k.github.io/srvdb/project/roadmap/) and
  [architecture decisions](docs/adr/)

## Community

Questions, feedback and project showcases live in
[GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions); see
the [Community page](https://srinivas26k.github.io/srvdb/project/community/).
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and report security
issues privately as described in [SECURITY.md](SECURITY.md).

## Contributing

Bug reports, docs, tests, benchmark results from your hardware and code are
all welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md); the engineering
rules every change follows are in [AGENTS.md](AGENTS.md).

## Contributors

Thank you to everyone who has helped build SrvDB. New contributors are very
welcome: pick an issue labelled
[`good first issue`](https://github.com/Srinivas26k/srvdb/labels/good%20first%20issue)
and follow [CONTRIBUTING.md](CONTRIBUTING.md).

<a href="https://github.com/Srinivas26k/srvdb/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Srinivas26k/srvdb" alt="SrvDB contributors">
</a>

Community pull requests under review:

- [@Safronix](https://github.com/Safronix): Python demo cleanup
  ([#74](https://github.com/Srinivas26k/srvdb/pull/74), for issue
  [#68](https://github.com/Srinivas26k/srvdb/issues/68))
- [@onlykushalll](https://github.com/onlykushalll): filtered search example
  ([#96](https://github.com/Srinivas26k/srvdb/pull/96), for issue
  [#66](https://github.com/Srinivas26k/srvdb/issues/66))

## Sponsor

SrvDB has no outside funding. Donations pay for CI on more platforms,
benchmark hardware and roadmap work:
[razorpay.me/@nampallisrinivas](https://razorpay.me/@nampallisrinivas) or the
[Sponsor page](https://srinivas26k.github.io/srvdb/project/sponsor/).
Commercial support and consulting: **srinivasvarma764@gmail.com**

## Cite SrvDB

If SrvDB or its benchmarks help your work, please cite the paper:

> Srinivas Nampalli. *The Index Is a Cache: Rethinking Vector Search for
> Edge AI.* Preprint, Zenodo, 2026.
> [doi:10.5281/zenodo.24568281](https://doi.org/10.5281/zenodo.24568281)

```bibtex
@misc{nampalli_2026_24568281,
  author    = {Nampalli, Srinivas},
  title     = {The Index Is a Cache: Rethinking Vector Search for Edge AI},
  month     = nov,
  year      = 2026,
  publisher = {Zenodo},
  version   = {v1},
  doi       = {10.5281/zenodo.24568281},
  url       = {https://doi.org/10.5281/zenodo.24568281}
}
```

GitHub's **Cite this repository** button (from `CITATION.cff`) gives the
same reference in other formats.

## License

[Apache License 2.0](LICENSE). Releases up to and including 0.2.0 were
published under AGPL-3.0. The SrvDB name and eagle logo are not covered by
the licence ([brand guide](assets/brand/README.md)).

## Acknowledgments

Built on [SimSIMD](https://github.com/ashvardanian/simsimd),
[Rayon](https://github.com/rayon-rs/rayon), [PyO3](https://github.com/PyO3/pyo3),
[redb](https://github.com/cberner/redb) and
[parking_lot](https://github.com/Amanieu/parking_lot).
