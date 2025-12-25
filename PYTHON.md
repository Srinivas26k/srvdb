# 🐍 SvDB Python Bindings

Python bindings for SvDB - the Zero-Gravity Embedded Vector Database.

## Installation

### Option 1: Install from PyPI (when published)
```bash
pip install svdb
```

### Option 2: Build from source
```bash
# Install maturin
pip install maturin

# Build and install
cd svdb
maturin develop --release

# Or build wheel
maturin build --release
```

## Quick Start

```python
import svdb

# Initialize database
db = svdb.SvDBPython("./my_vectors")

# Add vectors (ChromaDB-style API)
db.add(
    ids=["doc1", "doc2", "doc3"],
    embeddings=[
        [0.1] * 1536,  # 1536-dimensional vectors
        [0.2] * 1536,
        [0.3] * 1536
    ],
    metadatas=[
        '{"title": "Document 1"}',
        '{"title": "Document 2"}',
        '{"title": "Document 3"}'
    ]
)

# Search for similar vectors
results = db.search(query=[0.15] * 1536, k=10)
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")

# Get metadata
metadata = db.get("doc1")

# Persist to disk
db.persist()
```

## API Reference

### `SvDBPython(path: str)`
Initialize a new database instance.

**Args:**
- `path`: Directory path where database files will be stored

### `add(ids, embeddings, metadatas) -> int`
Add vectors in bulk (ChromaDB-style).

**Args:**
- `ids`: List of string IDs (must be unique)
- `embeddings`: List of vectors (each is a list of 1536 floats)
- `metadatas`: List of JSON strings for metadata

**Returns:**
- Number of vectors added

### `search(query, k) -> List[Tuple[str, float]]`
Search for similar vectors.

**Args:**
- `query`: Query vector (list of 1536 floats)
- `k`: Number of results to return

**Returns:**
- List of tuples `(id, score)` sorted by similarity

### `get(id) -> Optional[str]`
Get metadata for a specific ID.

**Args:**
- `id`: String ID of the vector

**Returns:**
- JSON metadata string or None

### `count() -> int`
Get the number of vectors in the database.

### `delete(ids) -> int`
Delete vectors by ID.

**Args:**
- `ids`: List of string IDs to delete

**Returns:**
- Number of vectors deleted

### `get_all_ids() -> List[str]`
Get all IDs in the database.

### `persist() -> None`
Persist all changes to disk.

## Performance

- **Search**: ~1ms for 10k vectors
- **Storage**: 192 bytes per vector (1536 dims)
- **Memory**: <50MB baseline (mmap-based)

## Examples

See `examples/python_demo.py` for a complete example.

## License

MIT OR Apache-2.0
