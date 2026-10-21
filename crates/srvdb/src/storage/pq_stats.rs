//! Compression statistics for product-quantized storage

use super::QuantizedVectorStorage;
use std::sync::atomic::Ordering;

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub vector_count: u64,
    pub memory_bytes: u64,
    pub full_precision_bytes: u64,
    pub compression_ratio: f32,
}

impl QuantizedVectorStorage {
    /// Get memory usage statistics
    pub fn get_stats(&self) -> StorageStats {
        let count = self.count.load(Ordering::Relaxed);
        let quantized_bytes = count * self.q_size as u64;
        let avg_dimension = self.q_size * self.quantizer.d_sub;
        let full_precision_bytes = count * (avg_dimension * std::mem::size_of::<f32>()) as u64;

        StorageStats {
            vector_count: count,
            memory_bytes: quantized_bytes,
            full_precision_bytes,
            compression_ratio: full_precision_bytes as f32 / quantized_bytes.max(1) as f32,
        }
    }
}
