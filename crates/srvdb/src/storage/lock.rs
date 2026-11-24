//! One process per database (ADR 0005)
//!
//! Opening a database takes an exclusive OS lock on `srvdb.lock` in its
//! directory and holds it until the `SrvDB` is dropped. The operating
//! system releases it when the process exits, including on a crash, so a
//! stale lock never blocks the next open.
//!
//! The lock is released explicitly on drop, not by closing the file. A
//! `flock` belongs to the open file description, which a child process
//! forked by any thread of the host shares until it calls exec; closing
//! our descriptor alone would leave the database locked for that window
//! (issue #82). Unlocking releases it for every copy at once.

use crate::error::Error;
use anyhow::{Context, Result};
use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

pub const FILE_NAME: &str = "srvdb.lock";

/// Held for the lifetime of an open database
#[derive(Debug)]
pub(crate) struct DbLock {
    file: File,
}

impl Drop for DbLock {
    fn drop(&mut self) {
        // Closing the file afterwards also releases the lock once no
        // forked child holds a copy, so a failure here is only logged.
        if let Err(e) = self.file.unlock() {
            log::warn!("could not release the database lock: {}", e.kind());
        }
    }
}

/// Take the exclusive lock for the database in `dir`.
///
/// # Errors
/// `Error::Locked` if another `SrvDB` (in this or another process) has the
/// database open.
pub(crate) fn acquire(dir: &Path) -> Result<DbLock> {
    let path = dir.join(FILE_NAME);
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(DbLock { file }),
        Err(TryLockError::WouldBlock) => Err(Error::Locked(format!(
            "{} is already open in another SrvDB instance or process; \
             close it first (one writer per database)",
            dir.display()
        ))
        .into()),
        Err(TryLockError::Error(e)) => {
            Err(e).with_context(|| format!("failed to lock {}", path.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn second_lock_is_refused_until_the_first_is_dropped() {
        let dir = TempDir::new().unwrap();
        let first = acquire(dir.path()).unwrap();
        let err = acquire(dir.path()).unwrap_err();
        assert!(matches!(Error::from(err), Error::Locked(_)));
        drop(first);
        assert!(acquire(dir.path()).is_ok());
    }
}
