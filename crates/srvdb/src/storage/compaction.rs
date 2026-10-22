//! File side of compaction (ADR 0004)
//!
//! `compact()` writes the compacted vector file as `<name>.compact`, then
//! commits the renumbered metadata in redb with a pending flag, then calls
//! [`finish`]. Opening a database calls [`finish`] when the flag is set (the
//! commit happened, the rename may not have) and [`discard_abandoned`]
//! otherwise (the commit never happened, so the new file is not wanted).

use super::atomic::sync_dir;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Vector files that can be compacted
const VECTOR_FILES: [&str; 3] = [
    super::flat::FILE_NAME,
    super::sq::FILE_NAME,
    super::pq::FILE_NAME,
];
/// Derived index files, rebuilt after compaction
const INDEX_FILES: [&str; 2] = ["hnsw.graph", super::ivf::FILE_NAME];

/// Where the compacted copy of `file` is written before the swap
pub(crate) fn staged(dir: &Path, file: &str) -> PathBuf {
    dir.join(format!("{file}.compact"))
}

/// Move compacted files into place and drop indexes that refer to old ids.
/// Safe to run more than once.
pub(crate) fn finish(dir: &Path) -> Result<()> {
    for file in VECTOR_FILES {
        let staged = staged(dir, file);
        if staged.exists() {
            std::fs::rename(&staged, dir.join(file))
                .with_context(|| format!("failed to finish compaction of {file}"))?;
        }
    }
    sync_dir(dir)?;
    for file in INDEX_FILES {
        let path = dir.join(file);
        if path.exists() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to remove stale {file}"))?;
        }
    }
    Ok(())
}

/// Remove staged files from a compaction that never committed
pub(crate) fn discard_abandoned(dir: &Path) -> Result<()> {
    for file in VECTOR_FILES {
        let staged = staged(dir, file);
        if staged.exists() {
            log::warn!("discarding unfinished compaction of {file}");
            std::fs::remove_file(&staged)?;
        }
    }
    Ok(())
}
