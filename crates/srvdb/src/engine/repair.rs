//! Explicit recovery of checksum sidecars (ADR 0007, issue #33)

use super::*;
use crate::quant::product::ProductQuantizer;
use crate::quant::rabitq::{self, SUPPORTED_BITS};
use crate::storage::header::VectorHeader;
use crate::storage::repair::{self, Outcome};
use crate::storage::{flat, pq, sq};

/// Record size of the vector file `name`, from its header (and, for PQ,
/// the codebooks); checks the header's mode byte against the file
fn record_size(dir: &Path, name: &str, header: &VectorHeader) -> Result<usize> {
    let dim = header.dimension as usize;
    let mode = header.quantization_mode;
    let wrong_mode = || {
        Error::Corrupt(format!(
            "{name} has quantization mode {mode}, which does not belong in that file; \
             restore the database from a backup"
        ))
    };
    match name {
        flat::FILE_NAME if mode == 0 => Ok(dim * std::mem::size_of::<f32>()),
        sq::FILE_NAME if mode == 1 => Ok(dim),
        pq::FILE_NAME if mode == 2 => {
            let bytes = std::fs::read(dir.join(pq::CODEBOOK_FILE)).map_err(|e| {
                Error::Corrupt(format!(
                    "{} cannot be read ({e}); the PQ record size is unknown, \
                     restore the database from a backup",
                    pq::CODEBOOK_FILE
                ))
            })?;
            let q = ProductQuantizer::from_bytes(&bytes)
                .map_err(|e| Error::Corrupt(format!("{}: {e:#}", pq::CODEBOOK_FILE)))?;
            if q.m * q.d_sub != dim {
                return Err(Error::Corrupt(format!(
                    "{} is for dimension {}, but {name} holds dimension {dim}",
                    pq::CODEBOOK_FILE,
                    q.m * q.d_sub
                )));
            }
            Ok(q.m)
        }
        crate::storage::rabitq::FILE_NAME
            if mode & 0xF0 == 0x10 && SUPPORTED_BITS.contains(&(mode & 0x0F)) =>
        {
            Ok(rabitq::record_size(dim, mode & 0x0F))
        }
        _ => Err(wrong_mode()),
    }
}

impl SrvDB {
    /// Recompute missing or damaged checksum files (`<file>.crc`) of the
    /// closed database at `path` from its vector files, so it opens again.
    ///
    /// Never runs implicitly; call it after open fails because a `.crc`
    /// file is missing or damaged (for example after copying only
    /// `vectors.bin`). For each vector file:
    ///
    /// - an intact `.crc` that matches the records is left alone;
    /// - an intact `.crc` that does **not** match means the records are
    ///   damaged: the call fails with `ChecksumMismatch` and writes nothing
    ///   for that file, so damaged data is never re-checksummed as good;
    /// - a missing `.crc`, or one that is damaged or describes another
    ///   file, is recomputed from the records as found.
    ///
    /// A rebuilt checksum cannot detect damage that was already in the
    /// records; they are trusted as they are. If the data itself may be
    /// damaged, restore from a backup instead. Deleting a `.crc` file and
    /// calling this blesses whatever the vector file holds.
    ///
    /// Durability: each sidecar is replaced atomically (temp file, fsync,
    /// rename, directory fsync). A crash leaves the old or the new sidecar;
    /// run the call again. Vector files, metadata and indexes are not
    /// changed. Holds the database lock while it runs.
    ///
    /// Returns the names of the checksum files that were rewritten (empty
    /// if every one was intact).
    ///
    /// # Errors
    /// `NotFound` if `path` is not a directory; `Locked` if the database is
    /// open; `ChecksumMismatch` if records do not match an intact checksum
    /// file; `Corrupt` if a vector file is truncated or its header (or the
    /// PQ codebooks) is invalid; I/O errors.
    pub fn rebuild_checksums(path: &str) -> Result<Vec<String>> {
        let dir = Path::new(path);
        if !dir.is_dir() {
            return Err(Error::NotFound(format!(
                "{} is not a database directory; pass the directory that holds vectors.bin",
                dir.display()
            )));
        }
        let _lock = crate::storage::lock::acquire(dir)?;
        let mut rebuilt = Vec::new();
        for name in [
            flat::FILE_NAME,
            sq::FILE_NAME,
            pq::FILE_NAME,
            crate::storage::rabitq::FILE_NAME,
        ] {
            let file = dir.join(name);
            if !file.is_file() {
                continue;
            }
            let header = repair::read_header(&file)?;
            let size = record_size(dir, name, &header)?;
            if repair::rebuild(&file, &header, size)? == Outcome::Rebuilt {
                rebuilt.push(format!("{name}.crc"));
            }
        }
        Ok(rebuilt)
    }
}
