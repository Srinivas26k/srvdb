//! Removing a database directory from disk (ADR 0013)
//!
//! Deletion first renames the directory, with its lock held, to a hidden
//! sibling tombstone (`.<name>.srvdb-deleting-<pid>`). From then on no other
//! process can open or lock the database at its old path, and the files are
//! removed from a directory nobody else knows about. On Unix the lock file
//! is unlinked before the lock is released, so it can never be re-taken.
//! On Windows the lock is released for the rename itself and re-taken in
//! the tombstone, because Windows cannot rename a directory holding an
//! open file (issue #98).

use super::*;
use crate::storage::identity;
use std::path::PathBuf;

/// Refuse a path that is a symlink, is not a directory, or holds anything
/// SrvDB did not write; return the entries
fn owned_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        // DirEntry::file_type does not follow symlinks
        if !entry.file_type()?.is_file() || !identity::is_owned(&name) {
            return Err(Error::invalid_input(format!(
                "{} contains '{name}', which SrvDB did not create; nothing was \
                 deleted. Remove the directory yourself if it is not needed",
                dir.display()
            )));
        }
        entries.push(entry.path());
    }
    Ok(entries)
}

/// The tombstone path next to `dir`
fn tombstone(dir: &Path) -> Result<PathBuf> {
    let name = dir.file_name().ok_or_else(|| {
        Error::invalid_input(format!(
            "{} does not name a directory; pass the database path itself",
            dir.display()
        ))
    })?;
    let parent = match dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    Ok(parent.join(format!(
        ".{}.srvdb-deleting-{}",
        name.to_string_lossy(),
        std::process::id()
    )))
}

/// Take the lock and move the database to `tomb`, returning the lock held
/// on the tombstone
#[cfg(not(windows))]
fn lock_and_move(dir: &Path, tomb: &Path) -> Result<storage::lock::DbLock> {
    let lock = storage::lock::acquire(dir)?;
    std::fs::rename(dir, tomb)?;
    Ok(lock)
}

/// Windows refuses to rename a directory while any file inside it is open
/// (`ERROR_ACCESS_DENIED`, issue #98), and our own lock handle is such a
/// file. So the lock is taken to prove the database is closed, released,
/// and taken again inside the tombstone after the rename. If another
/// process opens the database in that gap, its open handles make the
/// rename fail, which is reported as `Locked`; once the rename succeeds
/// the old path no longer exists, so nobody can open the database there.
#[cfg(windows)]
fn lock_and_move(dir: &Path, tomb: &Path) -> Result<storage::lock::DbLock> {
    drop(storage::lock::acquire(dir)?);
    match std::fs::rename(dir, tomb) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(Error::Locked(format!(
                "{} was opened by another SrvDB instance or process while it \
                 was being deleted; close it first (nothing was deleted)",
                dir.display()
            )))
        }
        Err(e) => return Err(e.into()),
    }
    match storage::lock::acquire(tomb) {
        Ok(lock) => Ok(lock),
        Err(e) => {
            std::fs::rename(tomb, dir)?;
            Err(e.into())
        }
    }
}

impl SrvDB {
    /// Delete the database at `path`: every SrvDB file, then the directory.
    ///
    /// Refuses anything that is not provably a SrvDB database, so a wrong
    /// path cannot remove unrelated data: `path` must be a real directory
    /// (not a symlink), contain only files SrvDB writes (no subdirectories,
    /// no other files) and at least one file with a SrvDB magic number. It
    /// also refuses while the database is open, in this or another process.
    ///
    /// With the lock held, the directory is first renamed to a hidden
    /// sibling `.<name>.srvdb-deleting-<pid>`, so nothing can open the
    /// database while its files are removed. A crash part way leaves that
    /// tombstone directory (or the database, if the crash came before the
    /// rename); remove a leftover tombstone by hand. Deleted data cannot be
    /// recovered.
    ///
    /// # Errors
    /// `NotFound` if `path` does not exist or is not a directory;
    /// `InvalidInput` if it is a symlink, holds files SrvDB did not write or
    /// no SrvDB file; `Locked` if the database is open; I/O.
    pub fn delete_database(path: &str) -> Result<()> {
        let dir = Path::new(path);
        let meta = match std::fs::symlink_metadata(dir) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::NotFound(format!(
                    "{} does not exist; nothing was deleted",
                    dir.display()
                )))
            }
            Err(e) => return Err(e.into()),
        };
        if meta.file_type().is_symlink() {
            return Err(Error::invalid_input(format!(
                "{} is a symbolic link; pass the database directory itself \
                 (nothing was deleted)",
                dir.display()
            )));
        }
        if !meta.is_dir() {
            return Err(Error::NotFound(format!(
                "{} is not a directory; nothing was deleted",
                dir.display()
            )));
        }
        owned_entries(dir)?;
        if !identity::has_magic(dir) {
            return Err(Error::invalid_input(format!(
                "{} is not a SrvDB database (no SrvDB file found); nothing was deleted",
                dir.display()
            )));
        }
        let tomb = tombstone(dir)?;
        let lock = lock_and_move(dir, &tomb)?;
        // Checked again inside the tombstone: the listing above ran before
        // the lock was held
        let entries = match owned_entries(&tomb) {
            Ok(entries) => entries,
            Err(e) => {
                std::fs::rename(&tomb, dir)?;
                return Err(e);
            }
        };
        let lock_file = tomb.join(storage::lock::FILE_NAME);
        for file in entries.iter().filter(|p| **p != lock_file) {
            std::fs::remove_file(file)?;
        }
        // Unix: unlink the lock while holding it, so it is never re-taken.
        // Windows cannot delete an open file: remove it after the drop (the
        // tombstone path is private, so nobody can take it in between).
        #[cfg(unix)]
        std::fs::remove_file(&lock_file)?;
        drop(lock);
        #[cfg(not(unix))]
        std::fs::remove_file(&lock_file)?;
        std::fs::remove_dir(&tomb)?;
        log::info!("deleted database ({} files)", entries.len());
        Ok(())
    }
}
