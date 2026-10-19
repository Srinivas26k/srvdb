//! The `SrvDB` engine: storage, indexes and metadata behind one API

mod commit;
mod constructors;
mod modes;
mod open;
mod read;
mod write;

#[cfg(test)]
mod tests;

use anyhow::Result;
use std::path::Path;

use crate::distance as search;
use crate::index::hnsw;
use crate::index::hybrid as hybrid_search;
use crate::index::ivf::{self, IVFIndex};
use crate::quant;
use crate::storage;
use crate::storage::metadata::MetadataStore;
use crate::storage::pq::{self as quantized_storage, QuantizedVectorStorage};
use crate::storage::{ScalarQuantizedStorage, VectorStorage};
use crate::tuning::auto_tune;
use crate::tuning::strategy::IndexMode;
use crate::types;
use crate::{SearchResult, Vector};


/// High-performance vector database trait
pub trait VectorEngine {
    fn new(path: &str) -> Result<Self>
    where
        Self: Sized;
    fn add(&mut self, vec: &Vector, meta: &str) -> Result<u64>;
    fn add_batch(&mut self, vecs: &[Vector], metas: &[String]) -> Result<Vec<u64>>;
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>>;
    fn search_batch(&self, queries: &[Vector], k: usize) -> Result<Vec<Vec<SearchResult>>>;
    fn get_metadata(&self, id: u64) -> Result<Option<String>>;
    fn persist(&mut self) -> Result<()>;
    fn count(&self) -> u64;
}

/// Main database implementation
pub struct SrvDB {
    pub(crate) path: std::path::PathBuf,
    pub(crate) vector_storage: Option<storage::flat::VectorStorage>,
    pub(crate) quantized_storage: Option<storage::pq::QuantizedVectorStorage>,
    pub(crate) scalar_storage: Option<storage::sq::ScalarQuantizedStorage>,
    pub(crate) metadata_store: MetadataStore,
    pub(crate) config: types::QuantizationConfig,
    pub(crate) index_type: types::IndexType,
    pub(crate) hnsw_index: Option<hnsw::HNSWIndex>,
    pub(crate) hnsw_config: Option<hnsw::HNSWConfig>,
    pub(crate) ivf_index: Option<ivf::IVFIndex>,
    pub(crate) ivf_config: Option<ivf::IVFConfig>,
    pub current_mode: IndexMode,
}

impl VectorEngine for SrvDB {
    fn new(path: &str) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        // Default to 1536 dimensions for backward compatibility
        let vector_storage = VectorStorage::new(path, 1536)?;
        let metadata_store = MetadataStore::new(path)?;

        Ok(Self {
            path: db_path.to_path_buf(),
            vector_storage: Some(vector_storage),
            quantized_storage: None,
            scalar_storage: None,
            metadata_store,
            config: types::QuantizationConfig::default(),
            index_type: types::IndexType::Flat,
            hnsw_index: None,
            hnsw_config: None,
            ivf_index: None,
            ivf_config: None,
            current_mode: IndexMode::Flat,
        })
    }

    fn add(&mut self, vec: &Vector, meta: &str) -> Result<u64> {
        self.insert_one(vec, meta)
    }

    fn add_batch(&mut self, vecs: &[Vector], metas: &[String]) -> Result<Vec<u64>> {
        self.insert_batch(vecs, metas)
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        self.search_vectors(query, k)
    }

    fn search_batch(&self, queries: &[Vector], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        self.search_vectors_batch(queries, k)
    }

    fn get_metadata(&self, id: u64) -> Result<Option<String>> {
        self.metadata_store.get(id)
    }

    fn persist(&mut self) -> Result<()> {
        self.commit()
    }

    fn count(&self) -> u64 {
        if let Some(ref scalar) = self.scalar_storage {
            scalar.count()
        } else if let Some(ref qstorage) = self.quantized_storage {
            qstorage.count()
        } else if let Some(ref vstorage) = self.vector_storage {
            vstorage.count()
        } else {
            0
        }
    }
}
