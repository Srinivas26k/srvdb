//! Conversion of the `where=` argument (a dict) into a core `Filter`

use crate::errors::to_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

/// Deepest nesting of dicts and lists accepted in `where` (the core's JSON
/// parser stops at 128; this also ends a walk over a self-referencing dict)
const MAX_DEPTH: usize = 64;

/// Serialise a `where` dict to JSON and let the core validate it.
///
/// Raises ValueError if it is not a dict, has a key that is not a str at
/// any level, nests deeper than 64, holds values JSON cannot encode
/// (including NaN), or is not a valid filter.
pub(crate) fn to_filter(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<srvdb::Filter> {
    if !value.is_instance_of::<PyDict>() {
        return Err(PyValueError::new_err(
            "where must be a dict such as {\"field\": value}",
        ));
    }
    check_keys(value, 0)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("allow_nan", false)?;
    // The exception text from json.dumps can quote the offending value, so
    // it is not passed on; the fixed message names only the problem type.
    let json: String = py
        .import("json")?
        .call_method("dumps", (value,), Some(&kwargs))
        .and_then(|s| s.extract())
        .map_err(|_| {
            PyValueError::new_err(
                "where must hold only strings, numbers, bools, None, lists and dicts \
                 (no NaN or infinity)",
            )
        })?;
    srvdb::Filter::from_json(&json).map_err(to_py)
}

/// json.dumps would silently turn int, float, bool and None keys into
/// strings, changing the field a filter names; reject them instead.
fn check_keys(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<()> {
    if depth > MAX_DEPTH {
        return Err(PyValueError::new_err(format!(
            "where nests dicts and lists deeper than {MAX_DEPTH} levels"
        )));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        for (key, item) in dict.iter() {
            if !key.is_instance_of::<PyString>() {
                return Err(PyValueError::new_err(
                    "where has a dict key that is not a str; field names and operators must be str",
                ));
            }
            check_keys(&item, depth + 1)?;
        }
    } else if let Ok(list) = value.cast::<PyList>() {
        for item in list.iter() {
            check_keys(&item, depth + 1)?;
        }
    } else if let Ok(tuple) = value.cast::<PyTuple>() {
        for item in tuple.iter() {
            check_keys(&item, depth + 1)?;
        }
    }
    Ok(())
}
