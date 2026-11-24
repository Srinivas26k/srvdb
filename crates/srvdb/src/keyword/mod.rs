//! BM25 keyword search over one metadata field, and hybrid ranking with
//! vector search through Reciprocal Rank Fusion (ADR 0015)
//!
//! The inverted index is derived data: `keyword.index` can always be
//! rebuilt from metadata. It is written by `persist()` after the metadata
//! commit point, so it may lag behind metadata but never lead it; on first
//! use after open it catches up (new records are tokenized, deleted ones
//! removed).

mod file;
mod fusion;
mod index;
mod maintain;
mod search;
mod tokenize;

pub(crate) use file::FILE_NAME;

use index::InvertedIndex;
use parking_lot::Mutex;

/// What the engine knows about its keyword index
#[derive(Debug, Default)]
enum Slot {
    /// `keyword.index` has not been read since open
    #[default]
    Unread,
    /// No keyword index is enabled
    Off,
    /// The index, and whether it differs from the file
    On { index: InvertedIndex, dirty: bool },
}

/// The keyword index held by a `SrvDB`. Behind a mutex because a search
/// (`&self`) may load it or catch it up with metadata.
#[derive(Debug, Default)]
pub(crate) struct KeywordState(Mutex<Slot>);

#[cfg(test)]
mod tests;
