//! Conversion of the `where=` argument (a dict) into a core `Filter`

use crate::errors::to_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Serialise a `where` dict to JSON and let the core validate it.
///
/// Raises ValueError if it is not a dict, holds values JSON cannot encode
/// (including NaN), or uses an operator outside the fixed set.
pub(crate) fn to_filter(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<srvdb::Filter> {
    if !value.is_instance_of::<PyDict>() {
        return Err(PyValueError::new_err(
            "where must be a dict such as {\"field\": value}",
        ));
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("allow_nan", false)?;
    let json: String = py
        .import("json")?
        .call_method("dumps", (value,), Some(&kwargs))
        .and_then(|s| s.extract())
        .map_err(|e| {
            PyValueError::new_err(format!(
                "where must hold only strings, numbers, bools, None, lists and dicts: {}",
                e.value(py)
            ))
        })?;
    srvdb::Filter::from_json(&json).map_err(to_py)
}
