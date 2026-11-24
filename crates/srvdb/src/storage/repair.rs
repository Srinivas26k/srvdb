//! Rebuilding a missing or damaged checksum sidecar (ADR 0007, issue #33)
//!
//! Only ever run on explicit request (`SrvDB::rebuild_checksums`), never on
//! open. The rules that keep a rebuild from blessing damaged data:
//!
//! - A sidecar that decodes (its own envelope CRC holds) and describes the
//!   file is trusted. If the records match it, nothing is written. If they
//!   do not, the records are damaged and the rebuild is refused with
//!   `Error::ChecksumMismatch`.
//! - An intact sidecar whose record size or block count disagrees with
//!   the header (or PQ codebooks) means the header is damaged: refused
//!   with `Error::Corrupt`, nothing written.
//! - Only a sidecar that is missing or fails its envelope check is
//!   recomputed, from the records as found. Damage that is already in those records
//!   cannot be detected and becomes part of the new checksums.
//!
//! The new table covers `min(header count, whole records present)`, the
//! same rule as the version 4 upgrade; `recover` then truncates to the
//! committed count on open. It is written with `write_atomic` (temp file,
//! fsync, rename, directory fsync), so a crash leaves either the old
//! sidecar or the new one, and the rebuild can simply be run again.

use super::checksum::{sidecar, BlockSums};
use super::header::VectorHeader;
use crate::error::Error;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Bytes read per step while checksumming (rounded down to whole records)
const READ_BYTES: usize = 1 << 20;

/// What `rebuild` did to one sidecar
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The sidecar was intact and matched; nothing was written
    Intact,
    /// The sidecar was missing or damaged and was recomputed
    Rebuilt,
    /// A version 4 file (no sidecar yet); open adds one
    Unversioned,
}

/// Read and validate the header of the vector file at `path`
pub(crate) fn read_header(path: &Path) -> Result<VectorHeader> {
    let mut bytes = [0u8; VectorHeader::SIZE];
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    match file.read_exact(&mut bytes) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(Error::Corrupt(format!(
                "{} is shorter than its {}-byte header; restore the database from a backup",
                file_name(path),
                VectorHeader::SIZE
            ))
            .into())
        }
        Err(e) => {
            return Err(e)
                .with_context(|| format!("failed to read the header of {}", path.display()))
        }
    }
    VectorHeader::from_bytes(&bytes)
}

/// Checksum the first `count` records of `path`, reading it sequentially
fn compute(path: &Path, record_size: usize, count: u64) -> Result<BlockSums> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    std::io::copy(
        &mut reader.by_ref().take(VectorHeader::SIZE as u64),
        &mut std::io::sink(),
    )?;
    let step = (READ_BYTES / record_size).max(1) * record_size;
    let mut buf = vec![0u8; step];
    let mut sums = BlockSums::new(record_size);
    let mut left = count * record_size as u64;
    while left > 0 {
        let n = (left.min(step as u64)) as usize;
        reader.read_exact(&mut buf[..n]).map_err(|_| {
            Error::Corrupt(format!(
                "{} holds fewer than {count} records; restore the database from a backup",
                path.display()
            ))
        })?;
        sums.append(&buf[..n])?;
        left -= n as u64;
    }
    Ok(sums)
}

/// Check or rebuild the sidecar of the vector file at `path`, whose
/// records are `record_size` bytes (see the module comment for the rules).
///
/// # Errors
/// `ChecksumMismatch` if an intact sidecar shows the records are damaged;
/// `Corrupt` if the vector file is truncated or its header is invalid;
/// I/O errors.
pub(crate) fn rebuild(path: &Path, header: &VectorHeader, record_size: usize) -> Result<Outcome> {
    if header.version < VectorHeader::VERSION_CHECKSUMS {
        return Ok(Outcome::Unversioned);
    }
    if record_size == 0 {
        return Err(Error::Corrupt(format!(
            "{} has a zero record size (dimension 0); restore the database from a backup",
            file_name(path)
        ))
        .into());
    }
    let side = sidecar(path);
    let bytes = match std::fs::read(&side) {
        Ok(bytes) => Some(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("failed to read {}", side.display())),
    };
    match bytes.map(|b| BlockSums::decode(&b)) {
        None => log::warn!("{} has no checksum file", file_name(path)),
        Some(Ok(stored)) => return check_intact(path, &stored, record_size),
        Some(Err(e)) => match Error::from(e) {
            Error::Corrupt(_) | Error::ChecksumMismatch(_) => {
                log::warn!("the checksum file of {} is damaged", file_name(path));
            }
            other => return Err(other.into()),
        },
    }
    let len = std::fs::metadata(path)?.len();
    let present = len.saturating_sub(VectorHeader::SIZE as u64) / record_size as u64;
    let count = header.count.min(present);
    let sums = compute(path, record_size, count)?;
    sums.save(path)?;
    log::warn!(
        "rebuilt the checksums of {} from its data ({count} records)",
        file_name(path)
    );
    Ok(Outcome::Rebuilt)
}

/// An intact sidecar is trusted: it must describe the file and match the
/// records, or the vector file (or the PQ codebooks) is what is damaged
fn check_intact(path: &Path, stored: &BlockSums, record_size: usize) -> Result<Outcome> {
    if !stored.describes(record_size) {
        return Err(Error::Corrupt(format!(
            "{}: header or codebooks disagree with an intact checksum file ({}); \
             restore a backup",
            file_name(path),
            file_name(&sidecar(path))
        ))
        .into());
    }
    if compute(path, record_size, stored.count())? == *stored {
        return Ok(Outcome::Intact);
    }
    Err(Error::ChecksumMismatch(format!(
        "{} is intact but the records of {} do not match it, so the data \
         is damaged; checksums were not rebuilt. Restore the database from \
         a backup",
        file_name(&sidecar(path)),
        file_name(path)
    ))
    .into())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}
