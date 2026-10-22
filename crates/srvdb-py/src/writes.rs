//! `Database` write methods: add, upsert and delete

use crate::database::{PyDatabase, State};
use crate::errors::to_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use srvdb::VectorEngine;
use std::collections::HashSet;

#[pymethods]
impl PyDatabase {
    /// Add new vectors.
    ///
    /// Args:
    ///     ids: unique, non-empty string ids (none may exist already)
    ///     embeddings: vectors matching the database dimension
    ///     metadatas: one JSON object string per vector
    ///
    /// Returns:
    ///     Number of vectors added
    ///
    /// Releases the GIL. Safe to call from several threads; concurrent
    /// calls on the same Database wait for each other.
    fn add(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Vec<String>,
    ) -> PyResult<usize> {
        let batch = Batch::new(ids, embeddings, metadatas)?;
        self.write(py, |s| {
            // Checked under the lock, so no other thread can add it meanwhile
            if let Some(id) = batch.ids.iter().find(|id| s.id_map.contains_key(*id)) {
                return Err(PyValueError::new_err(format!(
                    "id '{id}' already exists; use upsert() to replace it or delete() it first"
                )));
            }
            batch.store(s)
        })
    }

    /// Add new vectors and replace existing ones, by string id.
    ///
    /// An existing id afterwards refers to its new vector and metadata; the
    /// old vector is deleted. Arguments as for add().
    ///
    /// Returns:
    ///     Number of vectors written
    fn upsert(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Vec<String>,
    ) -> PyResult<usize> {
        let batch = Batch::new(ids, embeddings, metadatas)?;
        self.write(py, |s| batch.store(s))
    }

    /// Delete vectors by string id; unknown ids are ignored.
    ///
    /// Takes effect immediately and is made durable by persist().
    ///
    /// Returns:
    ///     Number of vectors deleted
    fn delete(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<usize> {
        self.write(py, |s| {
            let mut deleted = 0;
            for id in ids {
                if let Some(internal_id) = s.id_map.remove(&id) {
                    s.reverse_id_map.remove(&internal_id);
                    s.db.delete(internal_id).map_err(to_py)?;
                    deleted += 1;
                }
            }
            Ok(deleted)
        })
    }

    /// Remove deleted vectors from disk.
    ///
    /// Persists first. String ids are unaffected; search results are the
    /// same before and after. Rebuilds HNSW and IVF indexes, which takes time
    /// proportional to the data.
    ///
    /// Returns:
    ///     Number of vectors removed
    fn compact(&self, py: Python<'_>) -> PyResult<u64> {
        self.write(py, |s| {
            let removed = s.db.compact().map_err(to_py)?;
            s.load_id_maps()?;
            Ok(removed)
        })
    }
}

/// A validated batch, ready to store
struct Batch {
    ids: Vec<String>,
    vectors: Vec<srvdb::Vector>,
    metas: Vec<String>,
}

impl Batch {
    /// Validate lengths, ids and metadata (no lock needed)
    fn new(ids: Vec<String>, embeddings: Vec<Vec<f32>>, metadatas: Vec<String>) -> PyResult<Self> {
        let metas = prepare(&ids, &embeddings, &metadatas)?;
        let vectors = embeddings.into_iter().map(srvdb::Vector::new).collect();
        Ok(Self {
            ids,
            vectors,
            metas,
        })
    }

    /// Store the batch, then retire the vectors that existing ids pointed
    /// to. Runs under the exclusive lock.
    fn store(self, s: &mut State) -> PyResult<usize> {
        let internal_ids =
            VectorEngine::add_batch(&mut s.db, &self.vectors, &self.metas).map_err(to_py)?;
        for (id, internal_id) in self.ids.iter().zip(internal_ids) {
            if let Some(old) = s.id_map.insert(id.clone(), internal_id) {
                s.reverse_id_map.remove(&old);
                s.db.delete(old).map_err(to_py)?;
            }
            s.reverse_id_map.insert(internal_id, id.clone());
        }
        Ok(self.ids.len())
    }
}

/// Check lengths, ids and metadata of a batch; returns metadata with the
/// string id stored in it as `__id__`.
fn prepare(ids: &[String], embeddings: &[Vec<f32>], metadatas: &[String]) -> PyResult<Vec<String>> {
    if ids.len() != embeddings.len() || ids.len() != metadatas.len() {
        return Err(PyValueError::new_err(format!(
            "ids, embeddings and metadatas must have the same length (got {}, {} and {})",
            ids.len(),
            embeddings.len(),
            metadatas.len()
        )));
    }
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
    }
    ids.iter()
        .zip(metadatas)
        .enumerate()
        .map(|(i, (id, meta))| with_id(i, id, meta))
        .collect()
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
