//! Dynamic dimension vector storage for SrvDB v0.2.0
//! Supports 128-4096 dimensions with zero-copy mmap reads

use anyhow::{Context, Result};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::types::VectorHeader;

const BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB buffer

// Type alias for backward compatibility
pub type VectorStorage = DynamicVectorStorage;


pub struct DynamicVectorStorage {
    writer: BufWriter<File>,
    mmap: Option<MmapMut>,
    count: AtomicU64,
    dimension: usize,
    vector_size_bytes: usize,
    last_flushed_count: u64,
}

impl Drop for DynamicVectorStorage {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl DynamicVectorStorage {
    /// Create or open storage with specified dimension
    pub fn new(db_path: &str, dimension: usize) -> Result<Self> {
        let file_path = Path::new(db_path).join("vectors.bin");
        let exists = file_path.exists();
        
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_path)
            .context("Failed to open vectors.bin")?;

        let mut count = 0;
        let mut stored_dimension = dimension;

        if exists {
            // Read and validate existing header
            if file.metadata()?.len() >= VectorHeader::SIZE as u64 {
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                let header = unsafe { &*(mmap.as_ptr() as *const VectorHeader) };
                header.validate()?;
                
                count = header.count;
                stored_dimension = header.dimension as usize;
                
                // Ensure dimension matches
                if stored_dimension != dimension {
                    anyhow::bail!(
                        "Database dimension mismatch: expected {}, found {}. \
                        Cannot change dimension of existing database.",
                        dimension, stored_dimension
                    );
                }
            }
            file.seek(SeekFrom::End(0))?;
        } else {
            // Write new header
            let header = VectorHeader::new(dimension)?;
            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    &header as *const VectorHeader as *const u8,
                    VectorHeader::SIZE,
                )
            };
            file.write_all(header_bytes)?;
            file.sync_all()?;
        }

        let vector_size_bytes = dimension * std::mem::size_of::<f32>();
        let writer = BufWriter::with_capacity(BUFFER_SIZE, file);

        let file_ref = writer.get_ref();
        let mmap = if file_ref.metadata()?.len() > VectorHeader::SIZE as u64 {
            Some(unsafe { MmapOptions::new().map_mut(file_ref)? })
        } else {
            None
        };

        Ok(Self {
            writer,
            mmap,
            count: AtomicU64::new(count),
            dimension,
            vector_size_bytes,
            last_flushed_count: count,
        })
    }

    /// Append vector with zero-copy serialization
    #[inline]
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dimension {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            );
        }

        let id = self.count.fetch_add(1, Ordering::Relaxed);

        // Zero-copy write
        let vector_bytes = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const u8,
                self.vector_size_bytes,
            )
        };

        self.writer.write_all(vector_bytes)?;

        // Auto-flush when buffer is 90% full
        if self.writer.buffer().len() > (BUFFER_SIZE * 9 / 10) {
            self.writer.flush()?;
        }

        Ok(id)
    }

    /// Optimized batch append
    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let start_id = self.count.load(Ordering::Relaxed);
        let mut ids = Vec::with_capacity(vectors.len());

        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dimension {
                anyhow::bail!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i,
                    self.dimension,
                    vector.len()
                );
            }

            let vector_bytes = unsafe {
                std::slice::from_raw_parts(
                    vector.as_ptr() as *const u8,
                    self.vector_size_bytes,
                )
            };
            self.writer.write_all(vector_bytes)?;
            ids.push(start_id + i as u64);
        }

        self.count.store(start_id + vectors.len() as u64, Ordering::Relaxed);
        self.writer.flush()?;

        Ok(ids)
    }

    pub fn flush(&mut self) -> Result<()> {
        let current_count = self.count.load(Ordering::Relaxed);
        
        if current_count == self.last_flushed_count {
            return Ok(());
        }

        self.writer.flush()?;
        
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;
        
        let mut header = VectorHeader::new(self.dimension)?;
        header.count = current_count;
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const VectorHeader as *const u8,
                VectorHeader::SIZE,
            )
        };
        file.write_all(header_bytes)?;
        file.seek(SeekFrom::End(0))?;
        file.sync_all()?;
        
        drop(self.mmap.take());
        if file.metadata()?.len() > VectorHeader::SIZE as u64 {
            self.mmap = Some(unsafe { MmapOptions::new().map_mut(file as &File)? });
        }
        
        self.last_flushed_count = current_count;
        Ok(())
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Zero-copy vector access
    #[inline]
    pub fn get(&self, index: u64) -> Option<&[f32]> {
        if index >= self.count.load(Ordering::Relaxed) {
            return None;
        }
        
        let mmap = self.mmap.as_ref()?;
        let offset = VectorHeader::SIZE + (index as usize * self.vector_size_bytes);
        
        if offset + self.vector_size_bytes <= mmap.len() {
            let slice = &mmap[offset..offset + self.vector_size_bytes];
            Some(unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr() as *const f32,
                    self.dimension
                )
            })
        } else {
            None
        }
    }

    /// Get batch of vectors for SIMD processing
    pub fn get_batch(&self, start: u64, count: usize) -> Option<Vec<&[f32]>> {
        let end = start + count as u64;
        if end > self.count.load(Ordering::Relaxed) {
            return None;
        }

        let mut batch = Vec::with_capacity(count);
        for i in 0..count {
            if let Some(vec) = self.get(start + i as u64) {
                batch.push(vec);
            } else {
                return None;
            }
        }

        Some(batch)
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let count = self.count.load(Ordering::Relaxed) as usize;
        VectorHeader::SIZE + (count * self.vector_size_bytes)
    }
}

/// Scalar Quantized Storage (SQ8) - 4x compression
pub struct ScalarQuantizedStorage {
    storage: DynamicVectorStorage,
    quantizer: crate::types::ScalarQuantizer,
}

impl ScalarQuantizedStorage {
    /// Create new SQ8 storage with training data
    pub fn new_with_training(
        db_path: &str,
        dimension: usize,
        training_data: &[Vec<f32>],
    ) -> Result<Self> {
        if training_data.is_empty() {
            anyhow::bail!("Training data required for scalar quantization");
        }

        // Train quantizer
        let quantizer = crate::types::ScalarQuantizer::train(training_data)?;
        
        // Save quantizer
        let quantizer_path = Path::new(db_path).join("scalar_quantizer.bin");
        let quantizer_bytes = bincode::serialize(&quantizer)?;
        std::fs::write(quantizer_path, quantizer_bytes)?;

        // Note: For SQ8, we store uint8 vectors (dimension bytes per vector)
        // But we use DynamicVectorStorage with dimension for the original storage
        let storage = DynamicVectorStorage::new(db_path, dimension)?;

        Ok(Self { storage, quantizer })
    }

    /// Load existing SQ8 storage
    pub fn load(db_path: &str, dimension: usize) -> Result<Self> {
        let quantizer_path = Path::new(db_path).join("scalar_quantizer.bin");
        let quantizer_bytes = std::fs::read(quantizer_path)?;
        let quantizer = bincode::deserialize(&quantizer_bytes)?;

        let storage = DynamicVectorStorage::new(db_path, dimension)?;

        Ok(Self { storage, quantizer })
    }

    /// Append vector (automatically quantizes)
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        let encoded = self.quantizer.encode(vector);
        
        // Convert u8 to f32 for storage (temporary - will optimize later)
        let encoded_f32: Vec<f32> = encoded.iter().map(|&x| x as f32).collect();
        self.storage.append(&encoded_f32)
    }

    /// Get and decode vector
    pub fn get(&self, index: u64) -> Option<Vec<f32>> {
        let encoded_f32 = self.storage.get(index)?;
        let encoded: Vec<u8> = encoded_f32.iter().map(|&x| x as u8).collect();
        Some(self.quantizer.decode(&encoded))
    }

    /// Asymmetric search: compare full query to quantized vectors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        let count = self.storage.count() as usize;
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(count);
        
        for i in 0..count {
            if let Some(encoded_f32) = self.storage.get(i as u64) {
                let encoded: Vec<u8> = encoded_f32.iter().map(|&x| x as u8).collect();
                let score = self.quantizer.asymmetric_distance(query, &encoded);
                results.push((i as u64, score));
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    pub fn count(&self) -> u64 {
        self.storage.count()
    }

    pub fn flush(&mut self) -> Result<()> {
        self.storage.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_dynamic_dimensions() {
        let temp_dir = TempDir::new().unwrap();
        
        // Test 384-dim (MiniLM)
        let mut storage = DynamicVectorStorage::new(
            temp_dir.path().to_str().unwrap(),
            384
        ).unwrap();

        let vec = vec![0.5f32; 384];
        let id = storage.append(&vec).unwrap();
        assert_eq!(id, 0);

        storage.flush().unwrap();

        let retrieved = storage.get(0).unwrap();
        assert_eq!(retrieved.len(), 384);
        assert_eq!(retrieved[0], 0.5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = DynamicVectorStorage::new(
            temp_dir.path().to_str().unwrap(),
            768
        ).unwrap();

        let wrong_dim = vec![0.5f32; 384];
        assert!(storage.append(&wrong_dim).is_err());
    }

    #[test]
    fn test_reopen_dimension_check() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        {
            let mut storage = DynamicVectorStorage::new(path, 512).unwrap();
            storage.append(&vec![1.0; 512]).unwrap();
            storage.flush().unwrap();
        }

        // Reopen with same dimension - should work
        let storage = DynamicVectorStorage::new(path, 512).unwrap();
        assert_eq!(storage.count(), 1);

        // Try to open with different dimension - should fail
        let result = DynamicVectorStorage::new(path, 768);
        assert!(result.is_err());
    }

    #[test]
    fn test_scalar_quantization() {
        let temp_dir = TempDir::new().unwrap();
        
        let training: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![(i as f32) / 100.0; 256])
            .collect();

        let mut sq_storage = ScalarQuantizedStorage::new_with_training(
            temp_dir.path().to_str().unwrap(),
            256,
            &training,
        ).unwrap();

        let test_vec = vec![0.5; 256];
        let id = sq_storage.append(&test_vec).unwrap();
        sq_storage.flush().unwrap();

        let retrieved = sq_storage.get(id).unwrap();
        assert_eq!(retrieved.len(), 256);
        
        // Check reconstruction quality
        let error: f32 = test_vec.iter()
            .zip(retrieved.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        
        // Should have low reconstruction error
        assert!(error < 0.5);
    }
}