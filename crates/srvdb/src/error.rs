//! Typed errors returned by SrvDB's public API
//!
//! Messages describe what went wrong and how to fix it. They never contain
//! vector values or metadata contents.

use std::fmt;

/// Result type for SrvDB's public API
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by SrvDB
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The caller passed an argument that can never succeed as given
    /// (empty batch, `k = 0`, non-finite values, invalid JSON, ...).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// A vector's length does not match the database dimension.
    #[error("dimension mismatch: database has dimension {expected}, vector has {actual}")]
    DimensionMismatch {
        /// The database's dimension
        expected: usize,
        /// The length of the rejected vector
        actual: usize,
    },

    /// The operation is not valid in the database's current mode or state
    /// (for example training IVF before IVF mode is enabled).
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// The database is already open in another `SrvDB` instance, in this
    /// process or another one. Only one may have it open at a time.
    #[error("database is locked: {0}")]
    Locked(String),

    /// An on-disk file is missing, truncated or not a SrvDB file.
    #[error("corrupt or incompatible database file: {0}")]
    Corrupt(String),

    /// The operating system reported an I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// An unexpected internal failure. Please report it with the message.
    #[error("internal error: {0}")]
    Internal(String),
}

impl Error {
    pub(crate) fn invalid_input(msg: impl fmt::Display) -> Self {
        Self::InvalidInput(msg.to_string())
    }

    pub(crate) fn invalid_state(msg: impl fmt::Display) -> Self {
        Self::InvalidState(msg.to_string())
    }
}

/// Internal code uses `anyhow` for context chains. At the public boundary,
/// recover a typed error if one was raised inside, classify I/O failures,
/// and keep the full context chain in the message otherwise.
impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        let err = match err.downcast::<Error>() {
            Ok(typed) => return typed,
            Err(err) => err,
        };
        let message = format!("{err:#}");
        match err.downcast::<std::io::Error>() {
            Ok(io) => Self::Io(std::io::Error::new(io.kind(), message)),
            Err(_) => Self::Internal(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn typed_error_survives_anyhow_round_trip() {
        let inner: anyhow::Error = Error::invalid_input("k must be at least 1").into();
        let wrapped = inner.context("search failed");
        assert!(matches!(Error::from(wrapped), Error::InvalidInput(_)));
    }

    #[test]
    fn io_error_is_classified_with_context() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let err: anyhow::Error = Err::<(), _>(io)
            .context("Failed to open vectors.bin")
            .unwrap_err();
        match Error::from(err) {
            Error::Io(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
                assert!(e.to_string().contains("vectors.bin"));
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn other_errors_become_internal() {
        let err = anyhow::anyhow!("unexpected");
        assert!(matches!(Error::from(err), Error::Internal(_)));
    }
}
