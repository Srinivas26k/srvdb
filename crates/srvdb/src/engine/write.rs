//! Write path: append to storage, then update graph indexes

use super::*;

impl SrvDB {
    /// Insert freshly stored vectors into the active graph indexes (HNSW / IVF).
    ///
    /// Only full-precision storage is indexed; quantized-only databases have
    /// no f32 vectors to compute graph distances from.
    pub(super) fn index_new_ids(&mut self, ids: &[u64]) -> Result<()> {
        let ivf_trained = self
            .ivf_index
            .as_ref()
            .is_some_and(|ivf| !ivf.centroids.is_empty());
        if ids.is_empty() || (self.hnsw_index.is_none() && !ivf_trained) {
            return Ok(());
        }

        let Some(ref mut vstorage) = self.vector_storage else {
            return Ok(());
        };
        // Graph insertion reads vectors back through the mmap
        vstorage.make_visible()?;
        let vstorage = &*vstorage;

        let distance_fn = |a_id: u64, b_id: u64| -> f32 {
            if let (Some(a), Some(b)) = (vstorage.get(a_id), vstorage.get(b_id)) {
                1.0 - search::cosine_similarity(a, b)
            } else {
                f32::MAX
            }
        };

        if let Some(ref hnsw) = self.hnsw_index {
            for &id in ids {
                hnsw.insert(id, &distance_fn)?;
            }
        }

        if let (Some(ref ivf), true) = (&self.ivf_index, ivf_trained) {
            use rayon::prelude::*;
            ids.par_iter().try_for_each(|&id| -> Result<()> {
                if let Some(vec_data) = vstorage.get(id) {
                    ivf.add(id, &Vector::new(vec_data.to_vec()), &distance_fn)?;
                }
                Ok(())
            })?;
        }

        Ok(())
    }

    pub(super) fn insert_one(&mut self, vec: &Vector, meta: &str) -> Result<u64> {
        self.ensure_quantized_storage(std::slice::from_ref(&vec.data))?;

        let id = if let Some(ref mut scalar) = self.scalar_storage {
            scalar.append(&vec.data)?
        } else if self.config.enabled {
            if let Some(ref mut qstorage) = self.quantized_storage {
                qstorage.append(&vec.data)?
            } else {
                anyhow::bail!("Quantization enabled but quantized storage not initialized");
            }
        } else if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.append(&vec.data)?
        } else {
            anyhow::bail!("Vector storage not initialized");
        };

        self.index_new_ids(&[id])?;

        self.metadata_store.set(id, meta)?;
        Ok(id)
    }

    pub(super) fn insert_batch(&mut self, vecs: &[Vector], metas: &[String]) -> Result<Vec<u64>> {
        if vecs.len() != metas.len() {
            anyhow::bail!("Vectors and metadata counts must match");
        }

        // Convert all vectors to Vec<f32>
        let embedded: Vec<Vec<f32>> = vecs.iter().map(|v| v.data.clone()).collect();
        self.ensure_quantized_storage(&embedded)?;

        // Batch append vectors based on mode
        let ids = if let Some(ref mut scalar) = self.scalar_storage {
            scalar.append_batch(&embedded)?
        } else if self.config.enabled {
            if let Some(ref mut qstorage) = self.quantized_storage {
                qstorage.append_batch(&embedded)?
            } else {
                anyhow::bail!("Quantization enabled but quantized storage not initialized");
            }
        } else if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.append_batch(&embedded)?
        } else {
            anyhow::bail!("Vector storage not initialized");
        };

        self.index_new_ids(&ids)?;

        // Store metadata
        for (id, meta) in ids.iter().zip(metas.iter()) {
            self.metadata_store.set(*id, meta)?;
        }

        Ok(ids)
    }
}
