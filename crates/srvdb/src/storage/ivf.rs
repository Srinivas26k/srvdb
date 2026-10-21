//! IVF index persistence: one file, `ivf.index`, replaced atomically
//!
//! The configuration, centroids and every partition graph are saved
//! together, so after a crash a reader sees either the previous complete
//! index or the new one, never a mix of partitions from both. The index is
//! derived data (ADR 0003): if the file is missing it can be retrained from
//! the stored vectors.

use super::atomic::write_atomic;
use crate::error::Error;
use crate::hnsw::HNSWIndex;
use crate::ivf::{IVFConfig, IVFIndex};
use crate::Vector;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

pub const FILE_NAME: &str = "ivf.index";

#[derive(Serialize, Deserialize)]
struct SavedIvf {
    config: IVFConfig,
    centroids: Vec<Vector>,
    /// Each partition graph in its own serialized form
    partitions: Vec<Vec<u8>>,
}

pub struct IVFStorage;

impl IVFStorage {
    /// Save the whole index atomically to `db_path/ivf.index`
    pub fn save(index: &IVFIndex, db_path: &Path) -> Result<()> {
        let partitions = index
            .partitions
            .par_iter()
            .map(|partition| partition.read().to_bytes())
            .collect::<Result<Vec<_>>>()?;
        let saved = SavedIvf {
            config: index.config.clone(),
            centroids: index.centroids.clone(),
            partitions,
        };
        write_atomic(&db_path.join(FILE_NAME), &bincode::serialize(&saved)?)
    }

    /// Load the index saved by [`Self::save`], or `None` if there is none
    pub fn load(db_path: &Path) -> Result<Option<IVFIndex>> {
        let path = db_path.join(FILE_NAME);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).with_context(|| format!("failed to read {FILE_NAME}"))?;
        let saved: SavedIvf = bincode::deserialize(&bytes)
            .map_err(|e| Error::Corrupt(format!("{FILE_NAME}: {e}")))?;
        let partitions = saved
            .partitions
            .iter()
            .map(|bytes| Ok(Arc::new(RwLock::new(HNSWIndex::from_bytes(bytes)?))))
            .collect::<Result<Vec<_>>>()?;
        if partitions.len() != saved.config.nlist {
            return Err(Error::Corrupt(format!(
                "{FILE_NAME} has {} partitions but nlist is {}",
                partitions.len(),
                saved.config.nlist
            ))
            .into());
        }
        Ok(Some(IVFIndex {
            config: saved.config,
            centroids: saved.centroids,
            partitions,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn save_and_load_round_trip() {
        let dir = TempDir::new().unwrap();
        let config = IVFConfig {
            nlist: 2,
            ..Default::default()
        };
        let mut index = IVFIndex::new(config);
        let data: Vec<Vector> = (0..20)
            .map(|i| Vector::new(vec![i as f32, 1.0, (20 - i) as f32]))
            .collect();
        index.train(&data).unwrap();
        let distance = |a: u64, b: u64| (a as f32 - b as f32).abs();
        for (id, vector) in data.iter().enumerate() {
            index.add(id as u64, vector, &distance).unwrap();
        }

        IVFStorage::save(&index, dir.path()).unwrap();
        let loaded = IVFStorage::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.config.nlist, 2);
        assert_eq!(loaded.centroids.len(), 2);
        let sizes = |i: &IVFIndex| -> usize { i.partitions.iter().map(|p| p.read().len()).sum() };
        assert_eq!(sizes(&loaded), 20);
    }

    #[test]
    fn missing_file_loads_as_none() {
        let dir = TempDir::new().unwrap();
        assert!(IVFStorage::load(dir.path()).unwrap().is_none());
    }

    #[test]
    fn damaged_file_is_corrupt() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(FILE_NAME), b"not an index").unwrap();
        let err = IVFStorage::load(dir.path()).err().unwrap();
        assert!(matches!(Error::from(err), Error::Corrupt(_)));
    }
}
