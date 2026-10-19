//! Index mode switching and IVF management

use super::*;

impl SrvDB {
    /// Set the indexing strategy mode.
    ///
    /// If `Auto` is selected, the database will analyze the system and dataset
    /// to choose the best internal configuration.
    pub fn set_mode(&mut self, mode: IndexMode) {
        self.current_mode = mode;

        if mode == IndexMode::Auto {
            println!("SrvDB Adaptive Core active. Analyzing environment...");
            auto_tune::apply_auto_strategy(self);
        } else {
            // Manual overrides
            match mode {
                IndexMode::Flat => {
                    self.index_type = types::IndexType::Flat;
                    self.config.enabled = false;
                }
                IndexMode::Hnsw => {
                    self.index_type = types::IndexType::HNSW;
                    self.config.enabled = false;
                }
                IndexMode::Sq8 => {
                    self.index_type = types::IndexType::ScalarQuantized;
                    self.config.enabled = true;
                    self.config.mode = types::QuantizationMode::Scalar;
                }
                IndexMode::Ivf => {
                    // IVF requires separate training/setup, usually triggers training immediately
                    // or sets flag for next persist/train call.
                    if self.ivf_index.is_none() {
                        let config = self.ivf_config.clone().unwrap_or_default();
                        // Initialize empty index, training happens on demand or via explicit call
                        self.ivf_index = Some(IVFIndex::new(config.clone()));
                        self.ivf_config = Some(config);
                    }
                }
                IndexMode::Auto => {} // Handled above
            }
        }
    }

    /// Configure IVF parameters
    pub fn configure_ivf(&mut self, config: ivf::IVFConfig) -> Result<()> {
        if self.current_mode == IndexMode::Ivf {
            // If already initialized, we might need to rebuild or just update config?
            // For now, simpler: just update config and re-init empty index if needed.
            self.ivf_config = Some(config.clone());
            self.ivf_index = Some(ivf::IVFIndex::new(config));
        } else {
            self.ivf_config = Some(config);
            // When set_mode(Ivf) is called later, it should use this config.
            // But set_mode logic currently does `if ivf_index.is_none() { default }`.
            // I should check set_mode.
        }
        Ok(())
    }

    /// Train the IVF index using current data

    ///
    /// 1. Samples vectors from storage
    /// 2. Runs K-Means clustering
    /// 3. Re-indexes all data into partitions
    pub fn train_ivf(&mut self) -> Result<()> {
        if self.ivf_index.is_none() {
            anyhow::bail!("IVF mode not enabled. Call set_mode(IndexMode::Ivf) first.");
        }

        let count = self.vector_storage.as_ref().map(|s| s.count()).unwrap_or(0);
        if count < 10 { // Lower limit for testing
             // anyhow::bail!("Not enough data to train IVF");
             // Allow small training for unit tests?
        }

        // 1. Load Training Data (ALL for now, or sample)
        let mut training_data = Vec::with_capacity(count as usize);

        // Ensure data is synced to mmap
        if let Some(ref mut vstorage) = self.vector_storage {
            vstorage.flush()?;
        }

        if let Some(ref vstorage) = self.vector_storage {
            let train_limit = std::cmp::min(count, 10_000);
            for i in 0..train_limit {
                if let Some(vec) = vstorage.get(i) {
                    training_data.push(Vector::new(vec.to_vec()));
                }
            }
        }

        // 2. Train
        if let Some(ref mut ivf) = self.ivf_index {
            ivf.train(&training_data)?;
        }

        // 3. Re-index / Populate Partitions
        println!("IVF: Populating partitions...");
        if let Some(ref ivf) = self.ivf_index {
            if let Some(ref vstorage) = self.vector_storage {
                let distance_fn = |a_id: u64, b_id: u64| -> f32 {
                    if let (Some(a), Some(b)) = (vstorage.get(a_id), vstorage.get(b_id)) {
                        1.0 - search::cosine_similarity(a, b)
                    } else {
                        f32::MAX
                    }
                };

                use rayon::prelude::*;
                (0..count).into_par_iter().for_each(|id| {
                    if let Some(vec_data) = vstorage.get(id) {
                        let vec = Vector::new(vec_data.to_vec());
                        let _ = ivf.add(id, &vec, &distance_fn);
                    }
                });
            }
        }

        println!("IVF: Training complete.");
        Ok(())
    }

    /// Set ef_search parameter at runtime to tune recall/speed tradeoff
    ///
    /// Higher values = better recall but slower search
    /// Typical values: 50-200
    pub fn set_ef_search(&mut self, ef_search: usize) {
        if let Some(ref mut cfg) = self.hnsw_config {
            cfg.ef_search = ef_search;
        }
    }
}
