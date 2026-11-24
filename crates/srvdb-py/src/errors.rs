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
