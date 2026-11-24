//! Removing a database directory from disk (ADR 0013)

use super::*;
use crate::storage::identity;

impl SrvDB {
    /// Delete the database at `path`: every SrvDB file, then the directory.
    ///
    /// Refuses anything that is not provably a SrvDB database, so a wrong
    /// path cannot remove unrelated data: the directory must contain only
    /// files SrvDB writes (no subdirectories, no other files) and at least
    /// one file with a SrvDB magic number. It also refuses while the
    /// database is open, in this or another process.
    ///
    /// Not crash-atomic: a crash part way leaves some SrvDB files behind,
    /// and calling `delete_database` again finishes the job (as long as a
    /// file with a magic number is left; otherwise remove the directory by
    /// hand). Deleted data cannot be recovered.
    ///
    /// # Errors
    /// `NotFound` if `path` does not exist or is not a directory;
    /// `InvalidInput` if it holds files SrvDB did not write or no SrvDB
    /// file; `Locked` if the database is open; I/O.
    pub fn delete_database(path: &str) -> Result<()> {
        let dir = Path::new(path);
        if !dir.is_dir() {
            return Err(Error::NotFound(format!(
                "{} is not a directory; nothing was deleted",
                dir.display()
            )));
        }
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !entry.file_type()?.is_file() || !identity::is_owned(&name) {
                return Err(Error::invalid_input(format!(
                    "{} contains '{name}', which SrvDB did not create; nothing was \
                     deleted. Remove the directory yourself if it is not needed",
                    dir.display()
                )));
            }
            entries.push(entry.path());
        }
        if !identity::has_magic(dir) {
            return Err(Error::invalid_input(format!(
                "{} is not a SrvDB database (no SrvDB file found); nothing was deleted",
                dir.display()
            )));
        }
        let lock = storage::lock::acquire(dir)?;
        let lock_file = dir.join(storage::lock::FILE_NAME);
        for file in entries.iter().filter(|p| **p != lock_file) {
            std::fs::remove_file(file)?;
        }
        // The lock file goes last, after the lock is released (Windows
        // cannot delete an open file)
        drop(lock);
        match std::fs::remove_file(&lock_file) {
            Ok(()) => {}
            // acquire() created it if it was missing; nothing else to do
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        std::fs::remove_dir(dir)?;
        log::info!("deleted database ({} files)", entries.len());
        Ok(())
    }
}
