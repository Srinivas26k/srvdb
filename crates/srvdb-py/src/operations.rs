//! `Database` methods: insert, search, delete and maintenance

use crate::database::PyDatabase;
use crate::errors::to_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use srvdb::VectorEngine;
use std::collections::HashSet;

#[pymethods]
impl PyDatabase {
    /// Add vectors to database
    ///
    /// Args:
    ///     ids: List of string IDs
    ///     embeddings: List of vectors (each must match database dimension)
    ///     metadatas: List of JSON metadata strings
    ///
    /// Returns:
    ///     Number of vectors added
    fn add(
        &mut self,
        ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Vec<String>,
    ) -> PyResult<usize> {
        if ids.len() != embeddings.len() || ids.len() != metadatas.len() {
            return Err(PyValueError::new_err(format!(
                "ids, embeddings and metadatas must have the same length \
                 (got {}, {} and {})",
                ids.len(),
                embeddings.len(),
                metadatas.len()
            )));
        }

        // Ids must be non-empty and unique, in this batch and the database
        let mut seen = HashSet::with_capacity(ids.len());
        for (i, id) in ids.iter().enumerate() {
            if id.is_empty() {
                return Err(PyValueError::new_err(format!("id {i} is empty")));
            }
            if !seen.insert(id.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "duplicate id '{id}' within this batch"
                )));
            }
            if self.id_map.contains_key(id) {
                return Err(PyValueError::new_err(format!(
                    "id '{id}' already exists; use a new id or delete() it first"
                )));
            }
        }

        // Metadata must be a JSON object: the string id is stored in it
        let enriched_metas = ids
            .iter()
            .zip(&metadatas)
            .enumerate()
            .map(|(i, (id, meta))| with_id(i, id, meta))
            .collect::<PyResult<Vec<String>>>()?;

        let vectors: Vec<srvdb::Vector> = embeddings.into_iter().map(srvdb::Vector::new).collect();

        // Batch insert
        let internal_ids =
            VectorEngine::add_batch(&mut self.db, &vectors, &enriched_metas).map_err(to_py)?;

        // Update mappings
        for (i, internal_id) in internal_ids.iter().enumerate() {
            self.id_map.insert(ids[i].clone(), *internal_id);
            self.reverse_id_map.insert(*internal_id, ids[i].clone());
        }

        Ok(ids.len())
    }

    /// Search for k nearest neighbors
    ///
    /// Args:
    ///     query: Query vector (must match database dimension)
    ///     k: Number of results
    ///
    /// Returns:
    ///     List of (id, score) tuples sorted by score descending
    fn search(&mut self, query: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32)>> {
        let results = Python::attach(|py| {
            py.detach(|| VectorEngine::search(&self.db, &srvdb::Vector::new(query), k))
        })
        .map_err(to_py)?;

        Ok(results
            .into_iter()
            .filter_map(|r| {
                self.reverse_id_map
                    .get(&r.id)
                    .map(|id| (id.clone(), r.score))
            })
            .collect())
    }

    /// Batch search
    fn search_batch(
        &mut self,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> PyResult<Vec<Vec<(String, f32)>>> {
        let query_vecs: Vec<srvdb::Vector> = queries.into_iter().map(srvdb::Vector::new).collect();

        let results =
            Python::attach(|py| py.detach(|| VectorEngine::search_batch(&self.db, &query_vecs, k)))
                .map_err(to_py)?;

        Ok(results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .filter_map(|r| {
                        self.reverse_id_map
                            .get(&r.id)
                            .map(|id| (id.clone(), r.score))
                    })
                    .collect()
            })
            .collect())
    }

    fn get(&self, id: String) -> PyResult<Option<String>> {
        let internal_id = self
            .id_map
            .get(&id)
            .ok_or_else(|| PyValueError::new_err(format!("ID not found: {}", id)))?;

        VectorEngine::get_metadata(&self.db, *internal_id).map_err(to_py)
    }

    fn count(&self) -> usize {
        VectorEngine::count(&self.db) as usize
    }

    fn persist(&mut self) -> PyResult<()> {
        VectorEngine::persist(&mut self.db).map_err(to_py)
    }

    fn delete(&mut self, ids: Vec<String>) -> PyResult<usize> {
        let mut deleted = 0;
        for id in ids {
            if let Some(internal_id) = self.id_map.remove(&id) {
                self.reverse_id_map.remove(&internal_id);
                self.db.delete_metadata(internal_id).map_err(to_py)?;
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Get all IDs in the database
    fn get_all_ids(&self) -> PyResult<Vec<String>> {
        Ok(self.id_map.keys().cloned().collect())
    }

    /// Set index mode
    fn set_mode(&mut self, mode: String) -> PyResult<()> {
        match mode.to_lowercase().as_str() {
            "flat" => self.db.set_mode(srvdb::IndexMode::Flat),
            "hnsw" => self.db.set_mode(srvdb::IndexMode::Hnsw),
            "sq8" | "scalar" => self.db.set_mode(srvdb::IndexMode::Sq8),
            "ivf" => self.db.set_mode(srvdb::IndexMode::Ivf),
            "auto" => self.db.set_mode(srvdb::IndexMode::Auto),
            _ => return Err(PyValueError::new_err(format!("Invalid mode: {}", mode))),
        }
        self.index_type = mode;
        Ok(())
    }

    /// Configure IVF parameters
    ///
    /// Args:
    ///     nlist: Number of partitions (centroids)
    ///     nprobe: Number of partitions to search (search quality vs speed)
    #[pyo3(signature = (nlist=100, nprobe=10))]
    fn configure_ivf(&mut self, nlist: usize, nprobe: usize) -> PyResult<()> {
        let config = srvdb::ivf::IVFConfig {
            nlist,
            nprobe,
            max_iterations: 10,
            tolerance: 0.001,
            hnsw_config: Default::default(),
        };
        self.db.configure_ivf(config).map_err(to_py)
    }

    /// Train IVF index
    fn train_ivf(&mut self) -> PyResult<()> {
        // Python holds the GIL, but training is heavy. Release GIL?
        // Yes, allow threads.
        Python::attach(|py| py.detach(|| self.db.train_ivf())).map_err(to_py)
    }

    fn info(&self) -> PyResult<String> {
        let stats = if let Some(stats) = self.db.get_stats() {
            format!(
                "vectors={}, memory={}MB, compression={:.1}x",
                stats.vector_count,
                stats.memory_bytes / 1024 / 1024,
                stats.compression_ratio
            )
        } else {
            format!("vectors={}", self.count())
        };

        Ok(format!(
            "SrvDB(dimension={}, mode='{}', {})",
            self.dimension, self.index_type, stats
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        self.info()
    }
}

/// Add the string id to one metadata object as `__id__`.
fn with_id(index: usize, id: &str, meta: &str) -> PyResult<String> {
    let mut value: serde_json::Value = serde_json::from_str(meta)
        .map_err(|e| PyValueError::new_err(format!("metadata {index} is not valid JSON: {e}")))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| PyValueError::new_err(format!("metadata {index} must be a JSON object")))?;
    object.insert(
        "__id__".to_string(),
        serde_json::Value::String(id.to_string()),
    );
    Ok(value.to_string())
}
