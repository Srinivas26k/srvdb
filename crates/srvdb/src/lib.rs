//! # SrvDB
//!
//! An embedded vector database for local and offline AI applications.
//!
//! - Exact (flat) search, HNSW and IVF-HNSW approximate search
//! - Optional SQ8 (4x) and product quantization (PQ) compression
//! - Vectors in memory-mapped files, metadata in `redb`
//!
//! This crate is pure Rust. Python bindings live in the separate
//! `srvdb-py` crate.

mod config;
pub mod distance;
mod engine;
pub mod index;
pub mod quant;
pub mod storage;
pub mod tuning;
mod vector;

pub use config::{DatabaseConfig, IndexType, QuantizationConfig, QuantizationMode};
pub use engine::{SrvDB, VectorEngine};
pub use index::hnsw::{HNSWConfig, HNSWIndex};
pub use index::ivf::{IVFConfig, IVFIndex};
pub use quant::product::{ProductQuantizer, QuantizedVector};
pub use storage::metadata::MetadataStore;
pub use storage::{QuantizedVectorStorage, ScalarQuantizedStorage, VectorStorage};
pub use tuning::strategy::IndexMode;
pub use vector::{SearchResult, Vector};

/// Configuration and core types under their pre-0.3 path (`srvdb::types::...`)
pub mod types {
    pub use crate::config::*;
    pub use crate::quant::scalar::ScalarQuantizer;
    pub use crate::storage::header::VectorHeader;
    pub use crate::vector::*;
}

// Pre-0.3 module paths, kept so existing code keeps compiling
pub use index::{hnsw, hybrid as hybrid_search, ivf};
pub use tuning::{auto_tune, strategy};

// Short internal paths used across modules
pub(crate) use distance as search;
pub(crate) use storage::pq as quantized_storage;
