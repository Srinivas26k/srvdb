//! File side of compaction (ADR 0004)
//!
//! `compact()` writes the compacted vector file as `<name>.compact`, then
//! commits the renumbered metadata in redb with a pending flag, then calls
//! [`finish`]. Opening a database calls [`finish`] when the flag is set (the
//! commit happened, the rename may not have) and [`discard_abandoned`]
//! otherwise (the commit never happened, so the new file is not wanted).

use super::atomic::sync_dir;
use super::checksum::sidecar;
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
        // Both the records and their checksums (ADR 0007); finish runs to
        // the end before any vector file is opened, so their order is free
        let target = dir.join(file);
        for (from, to) in [
            (sidecar(&staged(dir, file)), sidecar(&target)),
            (staged(dir, file), target),
        ] {
            if from.exists() {
                std::fs::rename(&from, &to)
                    .with_context(|| format!("failed to finish compaction of {file}"))?;
            }
        }
    }
    sync_dir(dir)?;
    let rabitq = super::rabitq::ALL_FILES;
    for file in INDEX_FILES.iter().chain(rabitq.iter()) {
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
        for path in [sidecar(&staged), staged] {
            if path.exists() {
                log::warn!("discarding unfinished compaction of {file}");
                std::fs::remove_file(&path)?;
            }
        }
    }
    Ok(())
}
