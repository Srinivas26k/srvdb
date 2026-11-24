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
use srvdb::types::DatabaseConfig;
use srvdb::{SrvDB, Vector, VectorEngine};

fn main() -> anyhow::Result<()> {
    let config = DatabaseConfig::new(384)?;
    let mut db = SrvDB::new_with_config("./rust-db", config)?;

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
cheaper (with the default settings, under about 11% of 1M vectors or 35%
of 100k), SrvDB scores the matching ones exactly instead, which is faster
and exact. Other modes always score the matching vectors exactly. Filtered
search is Rust-only for now; from Python, filter the results of `search`.

## Notes

- Ids are `u64`, assigned in insertion order. `compact()` renumbers them.
- Other constructors: `SrvDB::new_with_hnsw`, `new_scalar_quantized`,
  `new_quantized` (PQ), `new_with_hnsw_quantized` (HNSW over PQ).
- Searches take `&self`; writes take `&mut self`. See
  [Threads and processes](/srvdb/guides/concurrency/).
- Errors are a typed `srvdb::Error`; see [Handling errors](/srvdb/guides/errors/).
- SrvDB logs through the [`log`](https://docs.rs/log) crate and never prints.
  Attach any logger (`env_logger`, `tracing-log`) to see its messages.
