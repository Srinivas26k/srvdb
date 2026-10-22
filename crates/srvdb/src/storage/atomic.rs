//! Crash-safe file replacement
//!
//! `write_atomic` leaves either the old file or the complete new file at
//! `path`, never a partial one: the bytes go to a temporary file in the same
//! directory, which is fsynced and then renamed over the target. On Unix the
//! directory is fsynced too so the rename itself survives power loss.

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    let name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?;
    let tmp = dir.join(format!(".{}.tmp", name.to_string_lossy()));

    let mut file =
        File::create(&tmp).with_context(|| format!("failed to create {}", tmp.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .with_context(|| format!("failed to write {}", tmp.display()))?;
    drop(file);

    fs::rename(&tmp, path).with_context(|| format!("failed to replace {}", path.display()))?;
    sync_dir(dir)
}

#[cfg(unix)]
pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .with_context(|| format!("failed to fsync directory {}", dir.display()))
}

#[cfg(not(unix))]
pub(crate) fn sync_dir(_dir: &Path) -> Result<()> {
    // Windows offers no directory fsync; MoveFileEx used by rename is
    // already durable once the call returns with the file flushed.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn replaces_contents_and_leaves_no_temp_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.bin");
        write_atomic(&path, b"first").unwrap();
        write_atomic(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        let names: Vec<_> = fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(names.len(), 1);
    }
}
