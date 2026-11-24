---
title: Rust API
description: The main types of the srvdb crate.
---

Generated API docs for every public item are on
[docs.rs/srvdb](https://docs.rs/srvdb) for released versions. For unreleased
code, run `cargo doc --open -p srvdb` in a clone.

| Item | Description |
|------|-------------|
| `SrvDB` | An open database. `Send + Sync`. `SrvDB::create(path, dimension, &options)` makes one, `SrvDB::open_existing(path)` reopens it, `SrvDB::delete_database(path)` removes it, `SrvDB::rebuild_checksums(path)` recomputes missing or damaged `.crc` files of a closed database. |
| `CreateOptions`, `IndexChoice` | Index chosen by name (`flat`, `hnsw`, `ivf`, `sq8`, `pq`, `rabitq`, `auto`) and its options, checked against that index, for `SrvDB::create`. |
| `VectorEngine` | Trait with `add`, `add_batch`, `search`, `search_batch`, `get_metadata`, `persist`, `count`. |
| `DatabaseConfig` | Dimension, index type and quantization for `SrvDB::new_with_config`. |
| `HNSWConfig`, `IVFConfig` | Index parameters. |
| `IndexMode` | `Flat`, `Hnsw`, `Sq8`, `Ivf`, `Auto`, for `set_mode`. |
| `Vector` | A vector of `f32` values. |
| `SearchResult` | `id: u64`, `score: f32` (cosine), `metadata: Option<String>`. |
| `Error`, `Result` | Typed errors; see [Handling errors](/srvdb/guides/errors/). |
| `Filter` | A metadata filter parsed by `Filter::from_json` (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`, `$not`) for `search_where`; see [Filtered search](/srvdb/guides/filtered-search/). |
| `Expansion`, `ExpandedResult` | Link expansion for `search_expand`: `Expansion::new(link_type, hops)` (1 to 3 hops) with `.with_max_results(n)` (default 100, at most 1,000); each `ExpandedResult` has `id`, `score`, `hop` and `metadata`. See [Links between records](/srvdb/guides/links/). |

`SrvDB` methods beyond the trait: `delete`, `replace`, `is_deleted`,
`live_count`, `compact`, `search_filtered`, `ids_where`, `search_where`, `search_batch_where`, `set_mode`, `set_ef_search`, `configure_ivf`,
`train_ivf`, `all_metadata`, `get_stats`, `index_name`, `dimension`,
`link`, `unlink`, `neighbors`, `link_count`, `move_links`, `search_expand`.

Example: [Using SrvDB from Rust](/srvdb/guides/rust/).
