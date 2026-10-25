//! # SrvDB Adaptive Strategy Engine
//!
//! The "Brain" of SrvDB. This module defines the decision logic for selecting the optimal
//! indexing strategy based on hardware constraints and data characteristics.
//!
//! "Choice is the enemy of speed." - We decide so the developer doesn't have to.

use serde::{Deserialize, Serialize};

/// High-level indexing mode selected by the user or the Auto-Tuner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IndexMode {
    /// Exact search (Brute Force). O(n).
    /// Best for small datasets (<50k) where 100% recall is critical.
    #[default]
    Flat,

    /// Approximate search (HNSW). O(log n).
    /// Best for real-time applications requiring sub-10ms latency.
    Hnsw,

    /// Scalar Quantization (SQ8). 4x Compression.
    /// Best for memory-constrained environments (e.g., Laptops, Edge).
    Sq8,

    /// Inverted File with HNSW Refinement (IVF-HNSW).
    /// Best for massive datasets (>1M vectors).
    Ivf,

    /// Choose the index automatically: flat for small datasets, HNSW for
    /// large ones. Never compresses vectors (see `tuning::auto_tune`).
    Auto,
}

/// Configuration for the Auto-Tuning Engine.
/// Defines thresholds for switching strategies.
#[derive(Debug, Clone)]
pub struct AutoTunerConfig {
    /// Total available system RAM in GB.
    pub available_ram_gb: f64,

    /// Dataset size from which auto mode builds an HNSW index.
    /// Default: 50,000 vectors.
    pub dataset_size_threshold: usize,

    /// Latency budget in milliseconds (P99).
    /// If projected latency > this, we prioritize HNSW.
    pub latency_budget_ms: f64,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            available_ram_gb: 4.0, // Conservative default
            dataset_size_threshold: 50_000,
            latency_budget_ms: 50.0,
        }
    }
}

impl AutoTunerConfig {
    /// Detect system hardware and return a config.
    /// Note: Actual detection happens in `auto_tune.rs` to keep this pure logic.
    pub fn new(ram_gb: f64) -> Self {
        Self {
            available_ram_gb: ram_gb,
            ..Default::default()
        }
    }
}

/// The index auto mode should use.
///
/// - Fewer vectors than `dataset_size_threshold`: `Flat`, exact search
///   that is fast enough at this size and has perfect recall.
/// - Larger, with at least 2 GB of RAM: `Hnsw`, approximate search in
///   logarithmic time.
/// - Larger but with less RAM: `Flat`, which reads vectors through the
///   memory map instead of holding a graph in memory.
///
/// Never returns `Sq8` or `Ivf`: compression changes stored data and IVF
/// needs training, so both are explicit choices.
pub fn decide_mode(config: &AutoTunerConfig, dataset_size: usize) -> IndexMode {
    if dataset_size < config.dataset_size_threshold || config.available_ram_gb < 2.0 {
        IndexMode::Flat
    } else {
        IndexMode::Hnsw
    }
}
