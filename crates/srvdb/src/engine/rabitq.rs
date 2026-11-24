//! RaBitQ mode (ADR 0008): full vectors in `vectors.bin`, low-bit codes
//! derived from them, searched by a code scan plus exact rerank

use super::*;
use crate::quant::rabitq::{RabitqConfig, MAX_DIMENSION as MAX_RABITQ_DIMENSION};
use crate::storage::rabitq::RabitqStorage;

/// Vectors encoded per append while catching codes up with storage
const ENCODE_CHUNK: u64 = 4096;

impl SrvDB {
    /// Create or open a database searched through RaBitQ codes.
    ///
    /// Vectors are stored at full precision in `vectors.bin` (the source
    /// of truth) and also encoded with `config.bits` bits per dimension
    /// after a random rotation; no training pass is needed. A search scans
    /// the codes for `k * config.rerank` candidates and rescores them
    /// exactly (see [`RabitqConfig`]). Codes are derived data: if they are
    /// missing, damaged or were made with other settings (bit width, seed)
    /// they are rebuilt from the vectors on open, which takes time
    /// proportional to the data.
    ///
    /// Thread-safety and durability are as for every mode (ADR 0003,
    /// ADR 0005): vectors are durable after `persist()`; codes are written
    /// after the commit point and caught up on the next open.
    ///
    /// # Errors
    /// `InvalidInput` for an unsupported bit width or a dimension of 0 or
    /// above [`crate::quant::rabitq::MAX_DIMENSION`] (4096), `DimensionMismatch` if `vectors.bin` holds another
    /// dimension, `Locked` if the database is open elsewhere, `Corrupt` /
    /// `ChecksumMismatch` for damaged vector or metadata files, or I/O.
    pub fn new_rabitq(path: &str, dimension: usize, config: RabitqConfig) -> Result<Self> {
        config.validate()?;
        if dimension == 0 || dimension > MAX_RABITQ_DIMENSION {
            return Err(Error::invalid_input(format!(
                "RaBitQ dimension must be between 1 and {MAX_RABITQ_DIMENSION}, got \
                 {dimension}; larger dimensions would need a rotation matrix too \
                 large to build on every open"
            )));
        }
        let db_path = Path::new(path);
        std::fs::create_dir_all(db_path)?;
        // First: opening metadata finishes an interrupted compaction
        let lock = storage::lock::acquire(db_path)?;
        let stored = settings::check(
            Path::new(path),
            dimension,
            &settings::Requested::Exact(storage::settings::IndexKind::Rabitq(config)),
        )?;
        let metadata_store = MetadataStore::new(path)?;
        let vector_storage = VectorStorage::new(path, dimension)?;

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
            current_mode: IndexMode::Flat,
            deleted: Default::default(),
            pending_deletes: Default::default(),
            keyword: Default::default(),
            _lock: lock,
            dimension,
        };
        db.recover()?;
        db.attach_rabitq(config)?;
        db.finish_settings(stored, true)?;
        Ok(db)
    }

    /// The RaBitQ settings in use, if this database searches RaBitQ codes
    pub fn rabitq_config(&self) -> Option<RabitqConfig> {
        self.rabitq_config
    }

    /// Change the rerank factor of a RaBitQ database (`0` = no rerank).
    /// Applies to the next search and is recorded by the next `persist()`
    /// (ADR 0011).
    ///
    /// # Errors
    /// `InvalidState` if the database is not in RaBitQ mode,
    /// `InvalidInput` if the factor is too large.
    pub fn set_rabitq_rerank(&mut self, rerank: usize) -> Result<()> {
        let Some(config) = self.rabitq_config else {
            return Err(Error::invalid_state(
                "not a RaBitQ database; open it with new_rabitq()",
            ));
        };
        let config = config.with_rerank(rerank);
        config.validate()?;
        self.rabitq_config = Some(config);
        Ok(())
    }

    /// Open the codes for `config`, or rebuild them from the vectors
    pub(crate) fn attach_rabitq(&mut self, config: RabitqConfig) -> Result<()> {
        // Closed first: on Windows mapped files cannot be removed
        self.rabitq = None;
        self.rabitq_config = Some(config);
        let storage = match RabitqStorage::open(&self.path, self.dimension, &config) {
            Ok(storage) => storage,
            Err(e) => {
                match Error::from(e) {
                    // No codes yet, or made with other settings
                    Error::InvalidState(why) => log::info!("building RaBitQ codes: {why}"),
                    // Damaged, or a code-file header of another width or
                    // dimension: derived data, so rebuild from the vectors
                    e @ (Error::Corrupt(_)
                    | Error::ChecksumMismatch(_)
                    | Error::InvalidInput(_)) => {
                        log::warn!("RaBitQ codes unusable, rebuilding them: {e}")
                    }
                    // I/O and other faults are not proof the codes are bad
                    other => return Err(other),
                }
                RabitqStorage::create(&self.path, self.dimension, &config)?
            }
        };
        self.rabitq = Some(storage);
        self.sync_rabitq()?;
        // Codes now cover exactly the committed vectors (recover() ran
        // first): make them durable so the next open does not re-encode
        if let Some(ref mut codes) = self.rabitq {
            codes.flush()?;
        }
        Ok(())
    }

    /// Make the codes cover exactly the stored vectors: drop codes beyond
    /// the vector count, encode vectors that have none. Idempotent.
    pub(super) fn sync_rabitq(&mut self) -> Result<()> {
        let (Some(codes), Some(vectors)) = (&mut self.rabitq, &self.vector_storage) else {
            return Ok(());
        };
        let count = vectors.count();
        if codes.count() > count {
            codes.truncate_to(count)?;
        }
        let mut next = codes.count();
        while next < count {
            let end = (next + ENCODE_CHUNK).min(count);
            let batch: Vec<&[f32]> = (next..end)
                .map(|id| {
                    vectors.get(id).ok_or_else(|| {
                        Error::Internal(format!("vector {id} vanished while encoding"))
                    })
                })
                .collect::<Result<_>>()?;
            codes.append(batch.into_iter())?;
            next = end;
        }
        Ok(())
    }

    /// Code scan, then exact rescoring of the candidates
    pub(super) fn search_rabitq(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        Ok(self.rabitq_top(query, k, None)?.0)
    }
}
