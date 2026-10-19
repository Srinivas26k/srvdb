//! On-disk storage: vector files, metadata and index persistence

pub mod flat;
pub mod header;
pub mod ivf;
pub mod metadata;
pub mod pq;
pub mod sq;

pub use flat::VectorStorage;
pub use pq::QuantizedVectorStorage;
pub use sq::ScalarQuantizedStorage;
