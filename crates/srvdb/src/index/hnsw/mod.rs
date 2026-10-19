//! Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search
//!
//! Based on: "Efficient and robust approximate nearest neighbor search using
//! Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)
//!
//! Key features:
//! - O(log n) search complexity instead of O(n)
//! - Thread-safe concurrent reads with RwLock
//! - Support for both full precision and quantized vectors
//! - Persistence to disk

mod graph;
#[cfg(test)]
mod tests;

use anyhow::Result;
use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

/// HNSW configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    /// Number of bidirectional links per node (M in paper)
    /// Typical values: 16-32
    /// Higher M = better recall, more memory
    pub m: usize,

    /// Maximum connections per node (M_max = M, M_max0 = M * 2 for layer 0)
    pub m_max: usize,
    pub m_max0: usize,

    /// Size of dynamic candidate list during construction (efConstruction in paper)
    /// Typical values: 200-500
    /// Higher ef_construction = better recall, slower construction
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search (ef in paper)
    /// Typical values: 50-200
    /// Higher ef_search = better recall, slower search
    pub ef_search: usize,

    /// Level multiplier for exponential decay (1/ln(M) in paper)
    pub ml: f32,

    /// Use with Product Quantization
    pub use_quantization: bool,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max: m,
            m_max0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f32).ln(),
            use_quantization: false,
        }
    }
}

impl HNSWConfig {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            m_max: m,
            m_max0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f32).ln(),
            use_quantization: false,
        }
    }
}

/// A node in the HNSW graph
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HNSWNode {
    /// Vector ID (index into storage)
    id: u64,

    /// Maximum layer this node appears in
    level: usize,

    /// Neighbors at each layer: layer -> [neighbor_ids]
    /// Layer 0 has the most connections (m_max0)
    /// Higher layers have fewer connections (m_max)
    neighbors: Vec<Vec<u64>>,
}

impl HNSWNode {
    fn new(id: u64, level: usize) -> Self {
        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }
        Self {
            id,
            level,
            neighbors,
        }
    }
}

/// Distance with ID for priority queue
#[derive(Debug, Clone, Copy)]
struct DistanceId {
    distance: f32,
    id: u64,
}

impl PartialEq for DistanceId {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for DistanceId {}

impl PartialOrd for DistanceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceId {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max-heap (BinaryHeap)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Distance function trait for flexibility
pub trait DistanceFunction: Send + Sync {
    fn distance(&self, a_id: u64, b_id: u64) -> f32;
    fn distance_to_query(&self, query_idx: usize, vec_id: u64) -> f32;
}

/// HNSW index for approximate nearest neighbor search
pub struct HNSWIndex {
    /// All nodes by ID
    nodes: Arc<RwLock<HashMap<u64, HNSWNode>>>,

    /// Entry point (top-level node with highest level)
    entry_point: Arc<RwLock<Option<u64>>>,

    /// Maximum level in the graph
    max_level: Arc<RwLock<usize>>,

    /// Configuration
    config: HNSWConfig,
}

impl HNSWIndex {
    /// Create a new HNSW index
    pub fn new(config: HNSWConfig) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            entry_point: Arc::new(RwLock::new(None)),
            max_level: Arc::new(RwLock::new(0)),
            config,
        }
    }

    /// Assign a random level to a new node using exponential decay
    /// P(level=l) = exp(-l * ln(M)) = (1/M)^l
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let uniform: f32 = rng.gen();
        let level = (-uniform.ln() * self.config.ml) as usize;
        level.min(16) // Cap at 16 levels
    }

    /// Get the number of nodes in the graph
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Serialize the HNSW graph to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let nodes = self.nodes.read();
        let entry_point = *self.entry_point.read();
        let max_level = *self.max_level.read();

        let data = (&*nodes, entry_point, max_level, &self.config);

        Ok(bincode::serialize(&data)?)
    }

    /// Deserialize the HNSW graph from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (nodes, entry_point, max_level, config): (
            HashMap<u64, HNSWNode>,
            Option<u64>,
            usize,
            HNSWConfig,
        ) = bincode::deserialize(bytes)?;

        Ok(Self {
            nodes: Arc::new(RwLock::new(nodes)),
            entry_point: Arc::new(RwLock::new(entry_point)),
            max_level: Arc::new(RwLock::new(max_level)),
            config,
        })
    }
}
