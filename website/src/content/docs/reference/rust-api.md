---
title: Rust API
description: The main types of the srvdb crate.
---

Full generated API docs (docs.rs) will be published when SrvDB is released on
crates.io. Until then, run `cargo doc --open -p srvdb` in a clone.

| Item | Description |
|------|-------------|
| `SrvDB` | An open database. `Send + Sync`. |
| `VectorEngine` | Trait with `add`, `add_batch`, `search`, `search_batch`, `get_metadata`, `persist`, `count`. |
| `DatabaseConfig` | Dimension, index type and quantization for `SrvDB::new_with_config`. |
| `HNSWConfig`, `IVFConfig` | Index parameters. |
| `IndexMode` | `Flat`, `Hnsw`, `Sq8`, `Ivf`, `Auto`, for `set_mode`. |
| `Vector` | A vector of `f32` values. |
| `SearchResult` | `id: u64`, `score: f32` (cosine), `metadata: Option<String>`. |
| `Error`, `Result` | Typed errors; see [Handling errors](/srvdb/guides/errors/). |

`SrvDB` methods beyond the trait: `delete`, `replace`, `is_deleted`,
`live_count`, `compact`, `set_mode`, `set_ef_search`, `configure_ivf`,
`train_ivf`, `all_metadata`, `get_stats`.

Example: [Using SrvDB from Rust](/srvdb/guides/rust/).
