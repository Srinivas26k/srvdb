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

## Notes

- Ids are `u64`, assigned in insertion order. `compact()` renumbers them.
- Other constructors: `SrvDB::new_with_hnsw`, `new_scalar_quantized`,
  `new_quantized` (PQ), `new_with_hnsw_quantized` (HNSW over PQ).
- Searches take `&self`; writes take `&mut self`. See
  [Threads and processes](/srvdb/guides/concurrency/).
- Errors are a typed `srvdb::Error`; see [Handling errors](/srvdb/guides/errors/).
- SrvDB logs through the [`log`](https://docs.rs/log) crate and never prints.
  Attach any logger (`env_logger`, `tracing-log`) to see its messages.
