//! Python bindings for SrvDB (`srvdb._low_level`)
//!
//! Only type conversion and error mapping live here; database logic is in
//! the `srvdb` crate.

mod database;
mod errors;
mod operations;

use database::PyDatabase;
use pyo3::prelude::*;

#[pymodule]
fn _low_level(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add("__version__", "0.2.0")?;
    m.add(
        "__doc__",
        "SrvDB v0.2.0 - Production Vector Database\n\n\
        Features:\n\
        - Dynamic dimensions (128-4096)\n\
        - Multiple index types (Flat, HNSW, SQ8, PQ)\n\
        - 4-32x compression with SQ8/PQ\n\
        - Sub-5ms latency\n\
        - <100MB memory for 10k vectors",
    )?;

    Ok(())
}
