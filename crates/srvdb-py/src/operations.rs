//! `Database` methods: insert, search, delete and maintenance

use crate::database::PyDatabase;
use crate::errors::to_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use srvdb::VectorEngine;

#[pymethods]
impl PyDatabase {
    /// Search for k nearest neighbors
    ///
    /// Args:
    ///     query: Query vector (must match database dimension)
    ///     k: Number of results
    ///
    /// Returns:
    ///     List of (id, score) tuples sorted by score descending
    ///
    /// Releases the GIL; several threads may search one Database at once.
    fn search(&self, py: Python<'_>, query: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32)>> {
        let query = srvdb::Vector::new(query);
        let results = py
            .detach(|| VectorEngine::search(&self.db, &query, k))
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

    /// Batch search; releases the GIL like search()
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> PyResult<Vec<Vec<(String, f32)>>> {
        let query_vecs: Vec<srvdb::Vector> = queries.into_iter().map(srvdb::Vector::new).collect();

        let results = py
            .detach(|| VectorEngine::search_batch(&self.db, &query_vecs, k))
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

    /// Number of searchable vectors (deleted ones are not counted)
    fn count(&self) -> usize {
        self.db.live_count() as usize
    }

    fn persist(&mut self) -> PyResult<()> {
        VectorEngine::persist(&mut self.db).map_err(to_py)
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
