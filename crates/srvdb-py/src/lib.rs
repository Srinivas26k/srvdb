//! Python bindings for SrvDB (`srvdb._low_level`)
//!
//! Only type conversion and error mapping live here; database logic is in
//! the `srvdb` crate.

mod constructors;
mod database;
mod errors;
mod graph;
mod lifecycle;
mod operations;
mod rabitq;
mod where_arg;
mod writes;

use database::PyDatabase;
use pyo3::prelude::*;

#[pymodule]
fn _low_level(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add_function(wrap_pyfunction!(lifecycle::create, m)?)?;
    m.add_function(wrap_pyfunction!(lifecycle::open, m)?)?;
    m.add_function(wrap_pyfunction!(lifecycle::delete, m)?)?;
    m.add_function(wrap_pyfunction!(lifecycle::rebuild_checksums, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "Compiled core of SrvDB, an embedded vector database for local and \
        offline AI. Import the public API from `srvdb`, not from this module.",
    )?;

    Ok(())
}
