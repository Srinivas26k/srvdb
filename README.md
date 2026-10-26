<p align="center">
  <a href="https://srinivas26k.github.io/srvdb/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/brand/png/srvdb-logo-horizontal-dark.png">
      <img src="assets/brand/png/srvdb-logo-horizontal.png" alt="SrvDB" height="110">
    </picture>
  </a>
</p>

<h3 align="center">The sharpest eye for your local AI.</h3>

<p align="center">
  An embedded vector database that runs inside your app, needs no server,<br>
  and keeps your data safe through crashes. Rust core, Python first.
</p>

<p align="center">
  <a href="https://pypi.org/project/srvdb/"><img src="https://img.shields.io/pypi/v/srvdb?color=F2B134&labelColor=1B2A4A" alt="PyPI"></a>
  <a href="https://pypi.org/project/srvdb/"><img src="https://img.shields.io/pypi/pyversions/srvdb?color=F2B134&labelColor=1B2A4A" alt="Python versions"></a>
  <a href="https://github.com/Srinivas26k/srvdb/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Srinivas26k/srvdb/ci.yml?branch=master&label=CI&labelColor=1B2A4A" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-C8102E?labelColor=1B2A4A" alt="License: Apache-2.0"></a>
</p>

<p align="center">
  <a href="https://srinivas26k.github.io/srvdb/"><b>Documentation</b></a> ·
  <a href="https://srinivas26k.github.io/srvdb/start/quickstart/">Quickstart</a> ·
  <a href="https://srinivas26k.github.io/srvdb/guides/local-rag/">Local RAG guide</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/Srinivas26k/srvdb/discussions">Discussions</a>
</p>

---

**SrvDB** (spelled S-R-V-D-B) is an open-source **embedded vector database**
for Python and Rust: a library for semantic search, RAG and AI memory that
stores embeddings on local disk, with no server and no cloud.

> **Not SurrealDB.** SrvDB is an independent project and is not related to
> SurrealDB, the multi-model database. Install SrvDB with `pip install srvdb`
> (PyPI) or `cargo add srvdb` (crates.io).

## Why SrvDB

- 🦅 **In-process, offline.** A library, not a service: no server, no Docker,
  no network access. `pip install srvdb` and go.
- 🛡️ **Crash-safe by design.** After `persist()` returns, your data survives a
  crash or power loss, and the database reopens exactly as it was. Every write
  path has a crash test.
- 🎯 **Honest results.** True cosine scores in [-1, 1] in every mode. Bad input
  (wrong dimension, NaN, duplicate ids, invalid metadata) is rejected before
  anything is written.
- ⚖️ **Your trade-off.** Exact search, HNSW, IVF, or 4× / up to 32× smaller
  storage with SQ8 and PQ: one argument to switch.
- 🧵 **Thread-safe.** Share one database between threads; every call releases
  the GIL and searches run in parallel.
- 🦀 **Rust inside.** SIMD distance kernels, typed errors, no Python runtime
  dependencies. Usable from Rust too.

## Use cases

- **Local RAG** over private documents with a local LLM
- **AI agent memory** that survives restarts: remember, recall, forget
- **Semantic search** inside desktop, CLI and notes apps
- **Offline and air-gapped** systems where data must stay on the machine
- **Edge devices** with 4× (SQ8) to 32× (PQ) smaller storage
- **Code, image and multimodal search**, deduplication, "more like this"
- **RAG tests in CI** without starting a database service

Details and when to choose something else:
[Use cases](https://srinivas26k.github.io/srvdb/start/use-cases/).

**100 projects in 20 industries**, five levels each, from beginner to
frontier. Cybersecurity alone goes from a SOC Copilot to Phishing Hunter,
the Wireshark-style PacketSense, a Threat Hunter Agent and an air-gapped OT
Sentinel. Also AML, fiscal auditing, GRC, aerospace, mechanical and
electrical engineering, quantum computing, energy and more: [Industries](https://srinivas26k.github.io/srvdb/industries/overview/).
Live examples and cheat sheets are coming soon.

## Install

```bash
pip install srvdb
```

Wheels for Python 3.9–3.14 on Linux, macOS and Windows. Building from source
needs Rust 1.99+ ([details](https://srinivas26k.github.io/srvdb/start/installation/)).

## 60-second quickstart

```python
import json
import srvdb

db = srvdb.Database("./my-db", dimension=384)    # any dimension 128–4096

db.add(
    ids=["intro", "pricing"],
    embeddings=[intro_vector, pricing_vector],   # lists of 384 floats
    metadatas=[json.dumps({"page": "intro"}), json.dumps({"page": "pricing"})],
)
db.persist()                                     # crash-safe from here on

for doc_id, score in db.search(question_vector, k=5):
    print(doc_id, round(score, 3), db.get(doc_id))

db.upsert(ids=["pricing"], embeddings=[new_vector], metadatas=['{"page": "pricing", "v": 2}'])
db.delete(["intro"])
db.persist()
```

## Local RAG in a few lines

```python
import json
import srvdb
from sentence_transformers import SentenceTransformer   # pip install sentence-transformers

model = SentenceTransformer("all-MiniLM-L6-v2")          # 384-dim, runs offline
db = srvdb.Database("./rag-db", dimension=384)

chunks = ["SrvDB stores vectors on local disk.", "persist() makes writes crash-safe."]
db.upsert(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    embeddings=model.encode(chunks).tolist(),
    metadatas=[json.dumps({"text": c}) for c in chunks],
)
db.persist()

question = model.encode("How do I make writes durable?").tolist()
context = [json.loads(db.get(i))["text"] for i, _ in db.search(question, k=2)]
```

Full walkthrough with chunking and a local LLM:
[Build a local RAG app](https://srinivas26k.github.io/srvdb/guides/local-rag/).

## Index modes

| Mode | Search | Storage per vector | Use when |
|------|--------|--------------------|----------|
| `flat` (default) | Exact | 4 B × dim | You want exact results |
| `hnsw` | Graph, approximate | vectors + graph | Search must be faster on large data |
| `ivf` | Partitions + graph | vectors + index | Very large collections |
| `sq8` | Scan, approximate | 1 B × dim (4×) | Storage matters |
| `pq` | Scan, approximate | dim ÷ 8 B (up to 32×) | Storage matters most |
| `auto` | Flat, then HNSW | vectors | You'd rather not choose |

Approximate modes trade accuracy for speed or size, depending on your data:
measure recall against `flat` before switching.
[How to choose](https://srinivas26k.github.io/srvdb/guides/index-modes/).

## The durability promise

> When `persist()` returns, every earlier `add`, `upsert` and `delete`
> survives a process crash or power loss. After a crash the database reopens
> in exactly the state of the last `persist()`: later writes are absent,
> never half-present.

How it works: [Durability](https://srinivas26k.github.io/srvdb/guides/durability/) ·
[ADR 0003](docs/adr/0003-durability.md).

## Performance

SrvDB publishes performance numbers only from a committed, reproducible
benchmark (hardware, OS, dataset, commit and command listed). That suite is
coming in 0.4; numbers from earlier versions were withdrawn because they
measured code with bugs fixed in 0.3. Measure on your own machine:

```bash
pip install "srvdb[bench]"
python benchmarks/universal_benchmark.py
```

## Use it from Rust

```rust
use srvdb::{types::DatabaseConfig, SrvDB, Vector, VectorEngine};

let mut db = SrvDB::new_with_config("./rust-db", DatabaseConfig::new(384)?)?;
db.add(&Vector::new(vec![0.1; 384]), r#"{"title": "first"}"#)?;
db.persist()?;
let hits = db.search(&Vector::new(vec![0.1; 384]), 5)?;
```

[Rust guide](https://srinivas26k.github.io/srvdb/guides/rust/)

## FAQ

**What is SrvDB?**
An embedded vector database: a Python and Rust library that stores
embeddings on local disk and finds the most similar ones to a query, inside
your own process. Use it for local RAG, semantic search and AI agent memory.

**Is SrvDB the same as SurrealDB?**
No. SrvDB is an embedded vector database library; SurrealDB is an unrelated
multi-model database server. The names only look alike.

**Does SrvDB need a server, Docker or internet access?**
No. It runs in-process, works fully offline and never opens a network
connection.

**How is SrvDB different from Chroma, FAISS or LanceDB?**
SrvDB focuses on a small, embeddable core with a documented crash-safety
guarantee: after `persist()` returns, data survives a crash or power loss.
FAISS is a similarity-search library that leaves metadata and transactional
storage to you; Chroma and LanceDB are broader data platforms.

**Which embedding models work?**
Any model with 128 to 4096 dimensions, for example sentence-transformers,
OpenAI, Cohere or Nomic embeddings.

## Roadmap

- **0.3 — Trustworthy storage** ✅ crash safety, delete/upsert/compact,
  validation, typed errors, thread safety, Apache-2.0
- **0.4 — Retrieval**: metadata filtering, hybrid search, LangChain and
  LlamaIndex integrations, reproducible benchmarks, research into
  training-free quantization

[Full roadmap](https://srinivas26k.github.io/srvdb/project/roadmap/)

## Contributing

Bug reports, docs, tests, benchmarks from your hardware and code are all
welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md); the engineering rules
every change follows are in [AGENTS.md](AGENTS.md). Questions and ideas:
[GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions).
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md); report security
issues privately as described in [SECURITY.md](SECURITY.md).

## License

SrvDB is licensed under the [Apache License 2.0](LICENSE): free to use, modify
and ship in open-source and commercial products, with an explicit patent
grant. Releases up to and including 0.2.0 were published under AGPL-3.0.
The SrvDB name and eagle logo are not covered by the licence
([brand guide](assets/brand/README.md)).

Commercial support and consulting: **srinivasvarma764@gmail.com**

## Acknowledgments

Built on [SimSIMD](https://github.com/ashvardanian/simsimd),
[Rayon](https://github.com/rayon-rs/rayon), [PyO3](https://github.com/PyO3/pyo3),
[redb](https://github.com/cberner/redb) and
[parking_lot](https://github.com/Amanieu/parking_lot).

<p align="center"><img src="assets/brand/srvdb-eagle.svg" alt="" width="48"></p>
