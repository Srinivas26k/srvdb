//! Mapping from `srvdb::Error` to Python exceptions

use pyo3::exceptions::{
    PyBlockingIOError, PyFileExistsError, PyFileNotFoundError, PyOSError, PyRuntimeError,
    PyValueError,
};
use pyo3::PyErr;
use srvdb::Error;

/// Convert a core error into the Python exception a caller should catch:
/// bad arguments and wrong mode are `ValueError`, operating-system I/O
/// failures are `OSError` (`FileNotFoundError` / `FileExistsError` for a
/// missing or already existing database, `BlockingIOError` when it is already
/// open elsewhere), anything else is `RuntimeError`.
/// A Python int that must be a count or size: a negative value is
/// `ValueError` naming the argument (pyo3's `usize` conversion would raise
/// `OverflowError`, which callers do not expect for bad input)
pub(crate) fn non_negative(name: &str, value: i64) -> Result<usize, PyErr> {
    usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be a non-negative integer, got {value}"
        ))
    })
}

pub(crate) fn to_py(err: Error) -> PyErr {
    let message = err.to_string();
    match err {
        Error::InvalidInput(_)
        | Error::DimensionMismatch { .. }
        | Error::InvalidState(_)
        | Error::SettingsMismatch(_) => PyValueError::new_err(message),
        Error::Io(_) => PyOSError::new_err(message),
        // OSError subclasses, as for os.makedirs / os.rmdir
        Error::NotFound(_) => PyFileNotFoundError::new_err(message),
        Error::AlreadyExists(_) => PyFileExistsError::new_err(message),
        // An OSError subclass: the database is open elsewhere
        Error::Locked(_) => PyBlockingIOError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}
