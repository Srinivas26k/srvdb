//! Read path: route a query to the active index and attach metadata

use super::*;

impl SrvDB {
    pub(super) fn search_vectors(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        validate::k(k)?;
        validate::vector(&query.data, self.dimension)?;
        let results: Vec<SearchResult> = if let Some(ref hnsw) = self.hnsw_index {
            // HNSW-accelerated search (O(log n))
            let raw_results = if self.config.enabled {
                if let Some(ref qstorage) = self.quantized_storage {
                    search::search_hnsw_quantized(qstorage, hnsw, &query.data, k, &self.deleted)?
                } else {
                    return Err(Error::invalid_state(
                        "Quantization enabled but quantized storage not initialized",
                    ));
                }
            } else if let Some(ref vstorage) = self.vector_storage {
                search::search_hnsw(vstorage, hnsw, &query.data, k, &self.deleted)?
            } else {
                return Err(Error::invalid_state("Vector storage not initialized"));
            };
            raw_results
                .into_iter()
                .map(|(id, score)| SearchResult::new(id, score, None))
                .collect()
        } else if let Some(ref ivf) = self.ivf_index {
            // IVF Search
            if ivf.centroids.is_empty() {
                // Fallback if not trained
                let raw_results = if let Some(ref vstorage) = self.vector_storage {
                    search::search_cosine(vstorage, &query.data, k, &self.deleted)?
                } else {
                    return Err(Error::invalid_state("Vector storage not initialized"));
                };
                raw_results
                    .into_iter()
                    .map(|(id, score)| SearchResult::new(id, score, None))
                    .collect()
            } else if let Some(ref vstorage) = self.vector_storage {
                // Partition graphs call distance_fn(query_id, candidate); the
                // first argument is a placeholder, distance is to the query.
                let distance_fn = |_query_id: u64, vec_id: u64| -> f32 {
                    if let Some(vec) = vstorage.get(vec_id) {
                        1.0 - search::cosine_similarity(&query.data, vec)
                    } else {
                        f32::MAX
                    }
                };
                let results = hybrid_search::search(ivf, query, k, &distance_fn, &self.deleted)?;
                // Hybrid search returns distances; convert to similarity
                results
                    .into_iter()
                    .map(|r| SearchResult::new(r.id, 1.0 - r.score, None))
                    .collect()
            } else {
                return Err(Error::invalid_state(
                    "Vector storage required for IVF search",
                ));
            }
        } else {
            // Flat search (O(n))
            let raw_results = if let Some(ref scalar) = self.scalar_storage {
                scalar.search(&query.data, k, &self.deleted)?
            } else if self.config.enabled {
                if let Some(ref qstorage) = self.quantized_storage {
                    search::search_quantized(qstorage, &query.data, k, &self.deleted)?
                } else {
                    // Quantizer not trained yet: nothing has been added
                    Vec::new()
                }
            } else if let Some(ref vstorage) = self.vector_storage {
                search::search_cosine(vstorage, &query.data, k, &self.deleted)?
            } else {
                return Err(Error::invalid_state("Vector storage not initialized"));
            };
            raw_results
                .into_iter()
                .map(|(id, score)| SearchResult::new(id, score, None))
                .collect()
        };

        // Enrich with metadata
        let enriched_results: Result<Vec<SearchResult>> = results
            .into_iter()
            .map(|mut res| {
                if let Ok(Some(meta)) = self.metadata_store.get(res.id) {
                    res.metadata = Some(meta);
                }
                Ok(res)
            })
            .collect();

        enriched_results
    }

    /// Search several queries in parallel, each exactly as `search_vectors`
    /// would (same index, tombstones and metadata).
    pub(super) fn search_vectors_batch(
        &self,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;
        validate::k(k)?;
        queries
            .par_iter()
            .map(|query| self.search_vectors(query, k))
            .collect()
    }
}
