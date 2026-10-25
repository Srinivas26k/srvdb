# Migration Guide

## Upgrading to v0.3

**Back up the database directory first.** Opening a 0.2 database with 0.3
upgrades `metadata.db` to a new file format, after which 0.2 can no longer
open it. Vector and quantizer files are read as they are; index files from
0.2 are rebuilt automatically on first open.

### Changes that may need action

| Change | What to do |
|--------|------------|
| Requires Python 3.9+ | Upgrade Python 3.8 environments. |
| `numpy`, `psutil`, `scikit-learn` no longer installed | Add them to your own dependencies if your code imports them. |
| Scores are true cosine similarity in [-1, 1] (were `(1 + cos) / 2` for flat, HNSW and IVF) | Re-tune any score threshold: new = 2 × old − 1. Result order is unchanged. |
| Metadata must be a JSON object | Wrap plain strings, e.g. `{"text": "..."}`. Invalid metadata now raises `ValueError` instead of being stored as `{}`. |
| Specific Python exceptions | `ValueError` for bad input, `OSError` for I/O, `BlockingIOError` if the database is open elsewhere, `RuntimeError` for internal faults. Code catching only `RuntimeError` must widen its `except`. |
| One open instance per database | Share one `Database` object between threads instead of opening the same path twice. |
| `persist()` is the commit point | Writes after the last `persist()` are discarded after a crash. Call `persist()` where you need durability. |
| `count()` excludes deleted vectors | Use it as the number of searchable vectors. |
| Auto mode never compresses | If you relied on auto switching to SQ8, open with `mode="sq8"` instead. |

### Python

The class is now `srvdb.Database`:

```python
import srvdb

db = srvdb.Database("./vectors", dimension=384)        # new
db = srvdb.SrvDBPython("./vectors", dimension=384)     # still works, warns
```

`srvdb.SrvDBPython` returns the same class and emits a
`DeprecationWarning`. It will be removed in v0.4.

New methods: `delete`, `upsert`, `compact`, `set_ef_search`.

### Rust

The crate is now split into `srvdb` (the engine) and `srvdb-py` (Python
bindings). The `srvdb::core::*` paths were removed; use `srvdb::types::*`
or the top-level re-exports (`srvdb::DatabaseConfig`, `srvdb::Vector`).
The public API returns `srvdb::Result` with a typed `srvdb::Error`
instead of `anyhow::Error`; match on its variants (`InvalidInput`,
`DimensionMismatch`, `InvalidState`, `Locked`, `Corrupt`, `Io`, ...).
`set_ef_search` now returns a `Result`. Building from source needs Rust
1.99 or newer.

## Upgrading from v0.1.x to v0.2.0

This guide helps you upgrade from SrvDB v0.1.x to v0.2.0. This release introduces significant architecture changes for better performance and flexibility.

## Major Changes

- **Renaming**: `SvDB` is now `SrvDB`. Python class is `SrvDBPython`, Rust struct is `SrvDB`.
- **Dynamic Dimension Support**: No longer hardcoded to 1536. You must specify dimensions explicitly.
- **Module Restructuring**: Code has been moved to cleaner domain-specific modules (`core`, `index`, `storage`, `api`).
- **New Index Types**: Added IVF (Hybrid) and SQ8 quantization support.

## Rust API Changes

### 1. Initialization
Old:
```rust
// v0.1.x (Hardcoded 1536 dimensions)
let mut db = SrvDB::new("/path/to/db").unwrap();
```

New:
```rust
// v0.2.0
use srvdb::core::types::DatabaseConfig;

let config = DatabaseConfig::new(1536)?; // Specify dimension
let mut db = srvdb::SrvDB::new_with_config("/path/to/db", config)?;
```

### 2. Module Imports
The crate exports have changed. Update your use statements:

| Old Path | New Path |
| :--- | :--- |
| `srvdb::Vector` | `srvdb::core::types::Vector` or `Vec<f32>` (raw) |
| `srvdb::IndexMode` | `srvdb::core::strategy::IndexMode` |
| `srvdb::quantization` | `srvdb::utils::quantization` |

### 3. Vector Type
We moved away from a custom `EmbeddedVector` to standard `Vec<f32>` for better interoperability.
```rust
// Adding data
let vec: Vec<f32> = vec![0.0; 1536]; // Standard vector
db.add(vec, meta)?;
```

## Python API Changes

### 1. Initialization
You must now provide the `dimension` argument if you are not using the default (1536 is default but explicit is better).

Old:
```python
# v0.1.x
db = srvdb.Database("./db")
```

New:
```python
# v0.2.0
# Support for 384, 768, 1536, etc.
db = srvdb.Database("./db", dimension=768)
```

### 2. New Modes
You can explicitly set modes like SQ8 or HNSW during init or runtime:

```python
# Create SQ8 compressed database (4x smaller)
db = srvdb.Database.new_scalar_quantized("./db", 768, training_vectors)

# Create HNSW index
db = srvdb.Database("./db", 768, mode="hnsw")
```

## Troubleshooting

- **"Dimension mismatch"**: Ensure the `dimension` passed to constructor matches your data.
- **"Vector storage not initialized"**: Some modes (like purely in-memory PQ) might behave differently, but `Flat`, `HNSW`, `SQ8` use disk persistence. Ensure you call `db.persist()` before terminating if you want to ensure data safety, though search usually auto-persists.

## Support
Open an issue on GitHub if you encounter problems upgrading.
