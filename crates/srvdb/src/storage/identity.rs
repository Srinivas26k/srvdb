//! Recognising a SrvDB database directory (ADR 0013)
//!
//! `SrvDB::create` refuses a directory that already holds files and
//! `SrvDB::delete_database` removes only directories that are provably
//! SrvDB databases: every entry is a file SrvDB writes, and at least one
//! file starts with a SrvDB magic number.

use super::{format, header::VectorHeader};
use std::io::Read;
use std::path::Path;

/// Every file SrvDB writes in a database directory
const OWNED: [&str; 12] = [
    super::flat::FILE_NAME,
    super::sq::FILE_NAME,
    super::sq::QUANTIZER_FILE,
    super::pq::FILE_NAME,
    super::pq::CODEBOOK_FILE,
    super::rabitq::FILE_NAME,
    super::rabitq::PARAMS_FILE,
    super::ivf::FILE_NAME,
    super::settings::FILE_NAME,
    super::lock::FILE_NAME,
    "hnsw.graph",
    "metadata.db",
];

/// Whether `name` is a file SrvDB writes: an owned file, its checksum
/// sidecar, a staged compaction copy or an atomic-write temp file
pub(crate) fn is_owned(name: &str) -> bool {
    let mut base = name;
    if let Some(inner) = base.strip_prefix('.').and_then(|n| n.strip_suffix(".tmp")) {
        base = inner;
    }
    for suffix in [".crc", ".compact"] {
        if let Some(inner) = base.strip_suffix(suffix) {
            base = inner;
        }
    }
    // `<file>.compact.crc` strips to `<file>.compact`, then to `<file>`
    if let Some(inner) = base.strip_suffix(".compact") {
        base = inner;
    }
    OWNED.contains(&base)
}

/// Whether `file` is a regular file (not a symlink, FIFO or device, which
/// could hang or redirect the read) that starts with `magic`
fn starts_with(file: &Path, magic: &[u8]) -> bool {
    if !std::fs::symlink_metadata(file).is_ok_and(|m| m.is_file()) {
        return false;
    }
    let mut head = vec![0u8; magic.len()];
    std::fs::File::open(file)
        .and_then(|mut f| f.read_exact(&mut head))
        .is_ok_and(|()| head == magic)
}

/// Whether `dir` holds a file with a SrvDB magic number: the settings
/// envelope (`SRVD`) or a vector file header (`SvDB`)
pub(crate) fn has_magic(dir: &Path) -> bool {
    let vector_magic = VectorHeader::MAGIC.to_le_bytes();
    starts_with(&dir.join(super::settings::FILE_NAME), format::MAGIC)
        || [
            super::flat::FILE_NAME,
            super::sq::FILE_NAME,
            super::pq::FILE_NAME,
        ]
        .iter()
        .any(|name| starts_with(&dir.join(name), &vector_magic))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_names_include_sidecars_and_temp_files() {
        for name in [
            "vectors.bin",
            "vectors.bin.crc",
            "vectors.bin.compact",
            "vectors.bin.compact.crc",
            ".hnsw.graph.tmp",
            ".metadata.db.tmp",
            "srvdb.lock",
        ] {
            assert!(is_owned(name), "{name}");
        }
        for name in ["notes.txt", "vectors.bin.bak", ".tmp", "photos"] {
            assert!(!is_owned(name), "{name}");
        }
    }
}
