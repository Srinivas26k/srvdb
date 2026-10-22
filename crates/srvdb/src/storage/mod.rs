//! On-disk storage: vector files, metadata and index persistence

pub(crate) mod atomic;
pub(crate) mod compaction;
pub mod flat;
pub(crate) mod format;
pub mod header;
pub mod ivf;
pub(crate) mod legacy;
pub mod metadata;
pub mod pq;
mod record_file;
pub mod sq;

pub use flat::VectorStorage;
pub use pq::QuantizedVectorStorage;
pub use sq::ScalarQuantizedStorage;
