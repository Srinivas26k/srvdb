//! Write path: append to storage, then update graph indexes

use super::*;

impl SrvDB {
    /// Insert freshly stored vectors into the active graph indexes (HNSW / IVF).
    ///
    /// Only full-precision storage is indexed; quantized-only databases have
    /// no f32 vectors to compute graph distances from.
    pub(super) fn index_new_ids(&mut self, ids: &[u64]) -> Result<()> {
        self.index_into(ids, true, true)
    }

    /// Insert `ids` into the HNSW graph and/or the trained IVF partitions
    pub(super) fn index_into(
        &mut self,
        ids: &[u64],
        into_hnsw: bool,
        into_ivf: bool,
    ) -> Result<()> {
        let into_hnsw = into_hnsw && self.hnsw_index.is_some();
        let into_ivf = into_ivf
            && self
                .ivf_index
                .as_ref()
                .is_some_and(|ivf| !ivf.centroids.is_empty());
        if ids.is_empty() || !(into_hnsw || into_ivf) {
            return Ok(());
        }

        // PQ storage has no full-precision vectors: build the graph from
        // the decoded (centroid) approximations, the same data search sees
        if self.vector_storage.is_none() {
            if let (Some(hnsw), Some(qstorage), true) =
                (&self.hnsw_index, &self.quantized_storage, into_hnsw)
            {
                let decoded = |id: u64| qstorage.get(id).and_then(|c| qstorage.quantizer.decode(c));
                let distance_fn = |a_id: u64, b_id: u64| -> f32 {
                    match (decoded(a_id), decoded(b_id)) {
                        (Some(a), Some(b)) => 1.0 - search::cosine_similarity(&a, &b),
                        _ => f32::MAX,
                    }
                };
                for &id in ids {
                    hnsw.insert(id, &distance_fn)?;
                }
            }
            return Ok(());
        }

        // Appended vectors are readable immediately (RecordFile)
        let Some(ref vstorage) = self.vector_storage else {
            return Ok(());
        };

        let distance_fn = |a_id: u64, b_id: u64| -> f32 {
            if let (Some(a), Some(b)) = (vstorage.get(a_id), vstorage.get(b_id)) {
                1.0 - search::cosine_similarity(a, b)
            } else {
                f32::MAX
            }
        };

        if let (Some(ref hnsw), true) = (&self.hnsw_index, into_hnsw) {
            for &id in ids {
                hnsw.insert(id, &distance_fn)?;
            }
        }

        if let (Some(ref ivf), true) = (&self.ivf_index, into_ivf) {
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
        validate::vector(&vec.data, self.dimension)?;
        validate::metadata(meta)?;
        self.ensure_quantized_storage(std::slice::from_ref(&vec.data))?;

        let id = if let Some(ref mut scalar) = self.scalar_storage {
            scalar.append(&vec.data)?
        } else if self.config.enabled {
            if let Some(ref mut qstorage) = self.quantized_storage {
                qstorage.append(&vec.data)?
            } else {
                return Err(Error::invalid_state(
                    "Quantization enabled but quantized storage not initialized",
                ));
            }
        } else if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.append(&vec.data)?
        } else {
            return Err(Error::invalid_state("Vector storage not initialized"));
        };

        self.index_new_ids(&[id])?;

        self.metadata_store.set(id, meta)?;
        Ok(id)
    }

    pub(super) fn insert_batch(&mut self, vecs: &[Vector], metas: &[String]) -> Result<Vec<u64>> {
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.data.as_slice()).collect();
        validate::batch(&slices, metas, self.dimension)?;
        if vecs.is_empty() {
            return Ok(Vec::new());
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
                return Err(Error::invalid_state(
                    "Quantization enabled but quantized storage not initialized",
                ));
            }
        } else if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.append_batch(&embedded)?
        } else {
            return Err(Error::invalid_state("Vector storage not initialized"));
        };

        self.index_new_ids(&ids)?;

        // Store metadata
        for (id, meta) in ids.iter().zip(metas.iter()) {
            self.metadata_store.set(*id, meta)?;
        }

        Ok(ids)
    }
}
