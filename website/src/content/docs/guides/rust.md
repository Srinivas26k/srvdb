---
title: Using SrvDB from Rust
description: Add SrvDB to a Rust project and store and search vectors.
---

```toml
[dependencies]
srvdb = { git = "https://github.com/Srinivas26k/srvdb" }
anyhow = "1"
```

```rust
use srvdb::{CreateOptions, IndexChoice, SrvDB, Vector, VectorEngine};

fn main() -> anyhow::Result<()> {
    let mut options = CreateOptions::new(IndexChoice::Hnsw);
    options.set("m", 16)?.set_exist_ok(true);
    let mut db = SrvDB::create("./rust-db", 384, &options)?;

    let id = db.add(&Vector::new(vec![0.1; 384]), r#"{"title": "first"}"#)?;
    db.persist()?;                                   // crash-safe from here

    for hit in db.search(&Vector::new(vec![0.1; 384]), 5)? {
        println!("{} {:.3} {:?}", hit.id, hit.score, hit.metadata);
    }

    db.delete(id)?;
    db.persist()?;
    Ok(())
}
```

## Filtered search

Search only the vectors whose metadata matches a condition. The filter is
applied during the search, so you get `k` results whenever at least `k`
vectors match (filtering the results of `search` afterwards can return
fewer).

```rust
use srvdb::{SrvDB, Vector};

fn recent(db: &SrvDB, query: &Vector) -> srvdb::Result<()> {
    // One pass over the metadata; reuse the set while the data is unchanged
    let allowed = db.ids_where(|meta| {
        serde_json::from_str::<serde_json::Value>(meta)
            .ok()
            .and_then(|v| v["year"].as_u64())
            .is_some_and(|year| year > 2020)
    })?;
    for hit in db.search_filtered(query, 5, &allowed)? {
        println!("{} {:.3}", hit.id, hit.score);
    }
    Ok(())
}
```

With an HNSW index, non-matching vectors are still walked through but
never returned. When few enough vectors match that scoring them directly is
cheaper (`matches² < 4 · ef_search · M · vectors`; see the
[measurements](https://github.com/Srinivas26k/srvdb/blob/master/docs/benchmarks.md#filtered-search)),
SrvDB scores the matching ones exactly instead, which is faster and exact. Other modes always score the matching vectors exactly. Filtered
search is Rust-only for now; from Python, filter the results of `search`.

## Notes

- Ids are `u64`, assigned in insertion order. `compact()` renumbers them.
- `SrvDB::create(path, dimension, &options)` makes a new database with an
  index chosen by name (`IndexChoice`) and options checked against that
  index; `SrvDB::open_existing(path)` reopens it with the recorded settings;
  `SrvDB::delete_database(path)` removes a closed database directory. The
  mode-specific constructors (`new_with_hnsw`, `new_scalar_quantized`,
  `new_quantized`, `new_with_hnsw_quantized`, `new_rabitq`) still work.
- Searches take `&self`; writes take `&mut self`. See
  [Threads and processes](/srvdb/guides/concurrency/).
- Errors are a typed `srvdb::Error`; see [Handling errors](/srvdb/guides/errors/).
- SrvDB logs through the [`log`](https://docs.rs/log) crate and never prints.
  Attach any logger (`env_logger`, `tracing-log`) to see its messages.
