# SrvDB: Embedded Vector Database for Offline AI Applications

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-AGPL-red.svg)](LICENSE)

SrvDB is a Rust-based embedded vector database designed for offline and edge AI deployments. It provides exact nearest neighbor search with a focus on simplicity, deterministic behavior, and zero external dependencies.

---

## Design Philosophy

**SrvDB is built for offline-first applications where:**
- Network connectivity cannot be assumed
- Deployment simplicity is critical
- Exact search results are required
- Resource constraints matter (edge devices, laptops, mobile)
- Zero-configuration deployment is preferred

**What SrvDB is NOT:**
- Not a distributed system (single-node by design)
- Not optimized for billion-scale datasets (target: 10k-1M vectors)
- Not a cloud service (fully embedded)

---

## Architecture

```
Storage Layer          Index Layer             API Layer
┌─────────────┐       ┌─────────────┐        ┌─────────────┐
│ vectors.bin │──────▶│ Flat Index  │───────▶│ Python API  │
│ (mmap)      │       │ (Exact)     │        │ (PyO3)      │
│             │       │             │        │             │
│ metadata.db │       │ HNSW Graph  │        │ Rust API    │
│ (redb)      │       │ (Approx)    │        │ (Native)    │
└─────────────┘       └─────────────┘        └─────────────┘
     ▲                      ▲                      ▲
     │                      │                      │
  SIMD Accel          Thread Safety         GIL-Free Search
(AVX-512/NEON)      (parking_lot)         (Concurrent)
```

---

## Performance Characteristics

### Measured Performance (M1 MacBook, 100k vectors, 1536-dim)

| Mode | Ingestion | Search Latency (P99) | Memory (RAM) | Disk Usage | Recall@10 |
|------|-----------|---------------------|--------------|------------|-----------|
| Flat | 23,979 vec/s | 11.2ms | 78MB | 594MB | 99.9% |
| HNSW | 23,562 vec/s | 10.6ms | 21MB | 594MB | 99.9% |
| HNSW+PQ | 4,613 vec/s | 3.5ms | -79MB* | 28MB | 13.4%** |

*Negative value indicates memory reclamation during PQ training  
**PQ recall degrades significantly on clustered semantic data

### Measured Performance (Google Colab, Various Dataset Sizes)

| Dataset Size | Insertion Time | Search (k=10) | Memory Increase | Disk Usage |
|--------------|----------------|---------------|-----------------|------------|
| 1,000 | 0.36s | 0.76ms | 1.3MB | 7.4MB |
| 10,000 | 2.17s | 6.00ms | 1.3MB | 60.1MB |
| 50,000 | 17.80s | 30.58ms | 2.8MB | 297.5MB |
| 100,000 | 41.18s | 65.91ms | 288.4MB | 594.5MB |

All tests maintain **100% Recall@10** with exact search.

---

## Installation

```bash
pip install srvdb
```

### Build from Source

```bash
git clone https://github.com/Srinivas26k/srvdb
cd srvdb
cargo build --release --features python
maturin develop --release
```

---

## Quick Start

```python
import srvdb
import numpy as np

# Initialize database
db = srvdb.SvDBPython("./vector_store")

# Prepare data (1536 dimensions required)
vectors = np.random.randn(10000, 1536).astype(np.float32)
ids = [f"doc_{i}" for i in range(10000)]
metadatas = [f'{{"index": {i}}}' for i in range(10000)]

# Bulk insert
db.add(
    ids=ids,
    embeddings=vectors.tolist(),
    metadatas=metadatas
)
db.persist()

# Search
query = np.random.randn(1536).astype(np.float32)
results = db.search(query=query.tolist(), k=10)

for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

---

## Supported Embedding Models

**Currently Supported Dimensions:** **1536 only**

### Compatible Models:
- OpenAI `text-embedding-ada-002` (1536-dim) ✅
- OpenAI `text-embedding-3-small` (1536-dim) ✅
- Nomic `nomic-embed-text-v1` (768-dim, 1536-dim) ✅ (use 1536 variant)
- Cohere `embed-english-v3.0` (1024-dim, **not supported**) ❌

### Unsupported Models:
- Sentence-Transformers `all-MiniLM-L6-v2` (384-dim) ❌
- Sentence-Transformers `all-mpnet-base-v2` (768-dim) ❌
- HuggingFace `BAAI/bge-small-en-v1.5` (384-dim) ❌

**Workaround for Non-1536 Embeddings:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
texts = ["sample text 1", "sample text 2"]
embeddings = model.encode(texts)  # Shape: (2, 384)

# Pad to 1536 dimensions
padded = np.pad(embeddings, ((0, 0), (0, 1536 - 384)), mode='constant')
padded = padded.astype(np.float32)

# Now compatible with SrvDB
db.add(
    ids=["id1", "id2"],
    embeddings=padded.tolist(),
    metadatas=['{"source": "text1"}', '{"source": "text2"}']
)
```

**Note:** Dimension flexibility is planned for v0.2.0.

---

## Indexing Modes

### 1. Flat Index (Exact Search - Default)

Brute-force linear scan with SIMD-accelerated cosine similarity.

```python
db = srvdb.SvDBPython("./db_flat")
```

**When to Use:**
- Datasets < 50,000 vectors
- 100% recall required
- Predictable latency needed
- Simplicity preferred

**Characteristics:**
- Time Complexity: O(n)
- Space Complexity: 6KB per vector (1536-dim × 4 bytes)
- Recall: 100% (exact)

### 2. HNSW Graph Index (Approximate Search)

Hierarchical Navigable Small World graph for O(log n) search.

```python
db = srvdb.SvDBPython.new_with_hnsw(
    "./db_hnsw",
    m=16,                  # Connections per node
    ef_construction=200,   # Build quality
    ef_search=50          # Search quality (tunable)
)

# Runtime tuning
db.set_ef_search(100)  # Higher = better recall, slower
```

**When to Use:**
- Datasets > 100,000 vectors
- Sub-millisecond latency required
- 95-99% recall acceptable

**Characteristics:**
- Time Complexity: O(log n)
- Space Complexity: ~6.2KB per vector (graph overhead: ~200 bytes)
- Recall: 95-99.9% (configurable via `ef_search`)

### 3. HNSW + Product Quantization (Memory-Efficient Hybrid)

Combines HNSW with 32x vector compression.

```python
# Prepare training data (5k-10k samples recommended)
training_vectors = vectors[:5000]

db = srvdb.SvDBPython.new_with_hnsw_quantized(
    "./db_hnsw_pq",
    training_vectors=training_vectors.tolist(),
    m=16,
    ef_construction=200,
    ef_search=50
)
```

**When to Use:**
- Memory-constrained environments (edge devices)
- Dataset > 500,000 vectors
- 85-95% recall acceptable
- **Uniform/random data distributions**

**Characteristics:**
- Time Complexity: O(log n)
- Space Complexity: ~392 bytes per vector (32x compression)
- Recall: **13-95%** (highly dependent on data distribution)

**⚠️ Critical Limitation:** PQ recall degrades severely on clustered/semantic data (e.g., document embeddings from the same topic). Measured recall: 13.4% on adversarial test dataset (70% random, 30% clustered). **Not recommended for RAG applications**.

---

## API Reference

### Initialization

```python
# Flat mode (default)
db = srvdb.SvDBPython(path: str)

# HNSW mode
db = srvdb.SvDBPython.new_with_hnsw(
    path: str,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50
)

# HNSW + PQ mode
db = srvdb.SvDBPython.new_with_hnsw_quantized(
    path: str,
    training_vectors: List[List[float]],
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50
)
```

### Operations

```python
# Insert vectors
db.add(
    ids: List[str],
    embeddings: List[List[float]],  # Each vector must be 1536-dim
    metadatas: List[str]            # JSON strings
) -> int

# Search
db.search(
    query: List[float],  # Must be 1536-dim
    k: int
) -> List[Tuple[str, float]]

# Batch search (parallel)
db.search_batch(
    queries: List[List[float]],
    k: int
) -> List[List[Tuple[str, float]]]

# Get metadata
db.get(id: str) -> Optional[str]

# Count vectors
db.count() -> int

# Persist to disk
db.persist() -> None

# HNSW runtime tuning
db.set_ef_search(ef: int) -> None  # HNSW mode only
```

---

## Use Cases

### 1. Retrieval-Augmented Generation (RAG)

```python
from openai import OpenAI

client = OpenAI()
db = srvdb.SvDBPython("./rag_index")

# Index documents
documents = ["...", "..."]
embeddings = client.embeddings.create(
    input=documents,
    model="text-embedding-ada-002"
).data

db.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=[e.embedding for e in embeddings],
    metadatas=[f'{{"text": "{doc}"}}' for doc in documents]
)
db.persist()

# Query
question = "What is machine learning?"
query_emb = client.embeddings.create(
    input=question,
    model="text-embedding-ada-002"
).data[0].embedding

results = db.search(query=query_emb, k=5)
```

**Recommendation:** Use Flat mode for RAG to guarantee recall. HNSW acceptable if dataset > 100k vectors.

### 2. Edge Device Deployment

```python
# Optimize for memory-constrained device
import sys

# Option 1: Flat mode with small dataset
if sys.platform == "linux":  # Raspberry Pi
    db = srvdb.SvDBPython("./edge_db")
    # Keep dataset < 10k vectors

# Option 2: HNSW+PQ (use with caution)
training_data = vectors[:2000]
db = srvdb.SvDBPython.new_with_hnsw_quantized(
    "./edge_db_pq",
    training_vectors=training_data.tolist(),
    m=8,  # Lower memory
    ef_construction=100,
    ef_search=20
)
```

### 3. Offline Semantic Search

```python
# Build index offline, deploy without network
db = srvdb.SvDBPython("./semantic_search")

# Index knowledge base
knowledge_base = load_documents()
embeddings = embed_model.encode(knowledge_base)

db.add(
    ids=[doc.id for doc in knowledge_base],
    embeddings=embeddings.tolist(),
    metadatas=[doc.metadata for doc in knowledge_base]
)
db.persist()

# Deploy: copy entire directory to target machine
# No network required for queries
```

---

## Performance Tuning

### HNSW Parameters

```python
# High accuracy (slower, more memory)
db = srvdb.SvDBPython.new_with_hnsw(
    path,
    m=32,              # More connections
    ef_construction=500,
    ef_search=200
)

# Balanced (recommended)
db = srvdb.SvDBPython.new_with_hnsw(
    path,
    m=16,
    ef_construction=200,
    ef_search=50
)

# Fast (lower accuracy)
db = srvdb.SvDBPython.new_with_hnsw(
    path,
    m=8,
    ef_construction=100,
    ef_search=20
)
```

### Environment Variables

```bash
# CPU-specific optimizations (build from source)
export RUSTFLAGS="-C target-cpu=native"
maturin build --release

# Buffer tuning
export SVDB_BUFFER_SIZE=8388608        # 8MB (default)
export SVDB_AUTO_FLUSH_THRESHOLD=1000  # Auto-flush every 1k vectors
```

---

## Benchmarking Your Hardware

We provide a universal benchmark script to validate performance on your specific hardware:

```bash
pip install srvdb numpy scikit-learn psutil
python universal_benchmark.py
```

The script:
- Automatically detects available RAM
- Adjusts dataset size accordingly (10k-1M vectors)
- Uses adversarial data mix (70% random, 30% clustered)
- Generates `benchmark_result_<os>_<timestamp>.json`

**Community Contribution:**  
Share your results in [GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions) to help validate performance across different CPUs (Intel, AMD, ARM/M1).

---

## Known Limitations

### Critical Limitations

1. **Fixed Dimensionality:** Only 1536-dimensional vectors supported in v0.1.8
   - **Workaround:** Pad smaller embeddings to 1536 dimensions
   - **Planned Fix:** v0.2.0 will support arbitrary dimensions

2. **Product Quantization Reliability:** PQ mode exhibits severe recall degradation (13-20%) on clustered semantic data
   - **Recommendation:** Use Flat or HNSW mode for RAG/semantic search
   - **PQ is safe for:** Uniformly distributed data (e.g., image embeddings, random features)

3. **Concurrent Write Contention:** Single-writer design (reads are concurrent)
   - Multiple processes cannot write simultaneously
   - **Workaround:** Use queue/coordinator for multi-process writes

### Minor Limitations

4. **No Dynamic Updates:** Vector deletion/update requires index rebuild
   - **Planned Fix:** v0.2.0 incremental updates

5. **Memory Measurement Artifacts:** Benchmark reports show inconsistent memory deltas
   - Actual memory usage is stable (verified via external profiling)
   - Measurement tooling improvement in progress

6. **HNSW Concurrency Scaling:** QPS plateaus beyond 4 threads in some configurations
   - Under investigation (potential lock contention in graph traversal)

---

## Future Work

### v0.2.0 Roadmap (Q1 2025)

**High Priority:**
- [ ] Variable dimension support (128-4096 dims)
- [ ] Incremental vector updates/deletion
- [ ] Async I/O for ingestion (target: 100k+ vec/s)
- [ ] Memory optimization (target: <100MB for 10k vectors)

**Medium Priority:**
- [ ] IVF-PQ indexing (alternative to HNSW for large datasets)
- [ ] Filtered search (metadata-based pre-filtering)
- [ ] GPU acceleration (CUDA/Metal for SIMD operations)
- [ ] Python type hints and `.pyi` stub files

**Low Priority:**
- [ ] Distributed sharding (multi-node deployment)
- [ ] Approximate quantization (scalar quantization, binary)
- [ ] Compression algorithms (Zstd for disk storage)

---

## Research Context

SrvDB's design is informed by the following research:

1. **HNSW Algorithm:**  
   Malkov, Y.A. & Yashunin, D.A. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI.

2. **Product Quantization:**  
   Jégou, H., Douze, M., & Schmid, C. (2011). *Product quantization for nearest neighbor search.* IEEE TPAMI.

3. **DiskANN:**  
   Subramanya, S.J., et al. (2019). *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS.

4. **Vector Database Survey:**  
   Wang, M., et al. (2021). *A Comprehensive Survey on Vector Database Management Systems.* VLDB.

---

## Contributing

Contributions are welcome in the following areas:

1. **Algorithm Improvements:**
   - Better PQ codebook training for semantic data
   - Alternative indexing structures (IVF, LSH)
   - GPU kernel optimization

2. **Engineering:**
   - Dimension flexibility implementation
   - Async I/O refactoring
   - Memory profiling and optimization

3. **Testing:**
   - Benchmark validation on diverse hardware
   - Integration tests with popular embedding models
   - Recall validation on real-world datasets

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

---

## License

GNU Affero General Public License v3.0

For commercial licensing inquiries, contact: srinivasvarma764@gmail.com

---

## Acknowledgments

SrvDB relies on:
- [SimSIMD](https://github.com/ashvardanian/simsimd) - SIMD distance kernels
- [Rayon](https://github.com/rayon-rs/rayon) - Data parallelism
- [PyO3](https://github.com/PyO3/pyo3) - Python-Rust bindings
- [redb](https://github.com/cberner/redb) - Embedded key-value store
- [parking_lot](https://github.com/Amanieu/parking_lot) - Fast synchronization primitives

---

## Citation

If you use SrvDB in academic research, please cite:

```bibtex
@software{srvdb2024,
  author = {Nampalli, Srinivas},
  title = {SrvDB: Embedded Vector Database for Offline AI Applications},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Srinivas26k/srvdb}
}
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/Srinivas26k/srvdb/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions)
- **Email:** srinivasvarma764@gmail.com

For bug reports, please include:
- SrvDB version (`pip show srvdb`)
- Python version
- Operating system and CPU architecture
- Minimal reproduction code
- Benchmark results (if performance-related)

---

**Status:** Production-ready for offline deployments with exact search requirements. HNSW and PQ modes should be validated on target data distribution before production use.
