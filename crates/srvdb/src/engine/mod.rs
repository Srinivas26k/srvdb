//! The `SrvDB` engine: storage, indexes and metadata behind one API

mod budget;
mod commit;
mod compact;
mod constructors;
mod create;
mod create_existing;
mod create_options;
mod delete;
mod destroy;
mod filter;
mod modes;
mod open;
mod open_existing;
mod rabitq;
mod rabitq_search;
mod read;
mod recover;
mod repair;
mod settings;
pub(crate) mod validate;
mod write;

#[cfg(test)]
mod tests;

pub use budget::MemoryReport;
pub use create_options::{CreateOptions, IndexChoice};

use crate::error::{Error, Result};
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
    /// Append one vector with its metadata; returns its id.
    ///
    /// With a memory budget (ADR 0018) the add is checked first: it may
    /// re-encode every stored vector to fewer bits, or fail with
    /// `Error::MemoryBudget` without writing anything. A re-encode is
    /// recorded in `index_settings.bin` before the vector is appended, so
    /// it stays in effect even if the append then fails.
    fn add(&mut self, vec: &Vector, meta: &str) -> Result<u64>;
    /// Append vectors with their metadata in one write; returns their ids.
    /// The memory-budget check and its re-encode work as for [`Self::add`],
    /// for the whole batch.
    fn add_batch(&mut self, vecs: &[Vector], metas: &[String]) -> Result<Vec<u64>>;
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>>;
    fn search_batch(&self, queries: &[Vector], k: usize) -> Result<Vec<Vec<SearchResult>>>;
    fn get_metadata(&self, id: u64) -> Result<Option<String>>;
    fn persist(&mut self) -> Result<()>;
    fn count(&self) -> u64;
}

/// An open SrvDB database.
///
/// # Concurrency (ADR 0005)
/// - One open instance per database directory: opening takes an exclusive
///   OS lock on `srvdb.lock`, held until this value is dropped. A second
///   open, in this process or another, fails with [`Error::Locked`].
/// - `SrvDB` is `Send + Sync`. Searches take `&self` and may run from many
///   threads at once (for example through an `Arc<SrvDB>`).
/// - Writes (`add`, `delete`, `persist`, `compact`, ...) take `&mut self`
///   and are exclusive; to mix readers and a writer across threads, wrap
///   the database in an `RwLock`.
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
    /// RaBitQ codes derived from `vector_storage` (ADR 0008)
    pub(crate) rabitq: Option<storage::rabitq::RabitqStorage>,
    pub(crate) rabitq_config: Option<quant::rabitq::RabitqConfig>,
    /// Contents of `index_settings.bin` as last read or written (ADR 0011);
    /// `None` for a database that has none yet (SrvDB 0.3 / early 0.4)
    pub(crate) stored_settings: Option<storage::settings::IndexSettings>,
    /// Memory budget in MiB (ADR 0018); writes step down the bit width
    /// or are refused to stay within it
    pub(crate) memory_budget: Option<u64>,
    pub current_mode: IndexMode,
    /// Vector dimension every insert and query must match
    pub(crate) dimension: usize,
    /// Tombstoned ids, skipped by every search (ADR 0004)
    pub(crate) deleted: std::collections::HashSet<u64>,
    /// Deletes since the last persist(), written by the next commit
    pub(crate) pending_deletes: Vec<u64>,
    /// Typed links between records (ADR 0016), primary data
    pub(crate) graph: crate::graph::LinkGraph,
    /// BM25 keyword index over one metadata field (ADR 0015)
    pub(crate) keyword: crate::keyword::KeywordState,
    /// Exclusive lock on the database directory, held while open
    pub(crate) _lock: storage::lock::DbLock,
}

impl VectorEngine for SrvDB {
    fn new(path: &str) -> Result<Self> {
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;

        // Default to 1536 dimensions for backward compatibility
        // First: opening metadata finishes an interrupted compaction
        let lock = storage::lock::acquire(Path::new(path))?;
        let fresh = settings::is_fresh(Path::new(path));
        let stored = settings::check(Path::new(path), 1536, &settings::Requested::Unspecified)?;
        let metadata_store = MetadataStore::new(path)?;
        let vector_storage = VectorStorage::new(path, 1536)?;

        let mut db = Self {
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
            rabitq: None,
            rabitq_config: None,
            stored_settings: None,
            memory_budget: None,
            current_mode: IndexMode::Flat,
            deleted: Default::default(),
            pending_deletes: Default::default(),
            graph: Default::default(),
            keyword: Default::default(),
            _lock: lock,
            dimension: 1536,
        };
        db.recover()?;
        db.finish_settings(stored, fresh)?;
        Ok(db)
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
        if self.deleted.contains(&id) {
            return Ok(None);
        }
        Ok(self.metadata_store.get(id)?)
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
