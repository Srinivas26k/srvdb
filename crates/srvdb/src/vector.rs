//! Vector and search result types

use serde::{Deserialize, Serialize};

/// Vector with dynamic dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub data: Vec<f32>,
}

impl Vector {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Normalize to unit length (L2 norm = 1.0)
    pub fn normalize(&mut self) {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in self.data.iter_mut() {
                *x /= norm;
            }
        }
    }
}

/// Search result with ID, similarity score, and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub metadata: Option<String>,
}

impl SearchResult {
    pub fn new(id: u64, score: f32, metadata: Option<String>) -> Self {
        Self {
            id,
            score,
            metadata,
        }
    }
}
