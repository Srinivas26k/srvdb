//! Full-precision (f32) vector storage in `vectors.bin`
//!
//! File handling and crash rules live in [`RecordFile`]; this type adds the
//! f32 view. Reads are zero-copy slices of the memory map.

use super::record_file::RecordFile;
use crate::error::Error;
use anyhow::Result;
use std::path::Path;

pub const FILE_NAME: &str = "vectors.bin";

// Type alias for backward compatibility
pub type VectorStorage = DynamicVectorStorage;

pub struct DynamicVectorStorage {
    file: RecordFile,
    dimension: usize,
}

impl DynamicVectorStorage {
    /// Create or open `vectors.bin` in `db_path` with the given dimension
    pub fn new(db_path: &str, dimension: usize) -> Result<Self> {
        let path = Path::new(db_path).join(FILE_NAME);
        let record_size = dimension * std::mem::size_of::<f32>();
        Ok(Self {
            file: RecordFile::open(&path, dimension, record_size, 0)?,
            dimension,
        })
    }

    fn check(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            }
            .into());
        }
        Ok(())
    }

    /// Append one vector; it is searchable as soon as this returns.
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        self.check(vector)?;
        self.file.append(bytemuck::cast_slice(vector))
    }

    /// Append vectors in one write; returns their ids.
    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let mut bytes = Vec::with_capacity(vectors.len() * self.file.record_size());
        for vector in vectors {
            self.check(vector)?;
            bytes.extend_from_slice(bytemuck::cast_slice(vector));
        }
        let first = self.file.append(&bytes)?;
        Ok((first..first + vectors.len() as u64).collect())
    }

    /// Make everything appended so far durable (fsync). Cheap when nothing
    /// changed since the last call.
    pub fn flush(&mut self) -> Result<()> {
        self.file.sync()
    }

    /// Write a compacted copy holding only `keep` (see storage::compaction)
    pub(crate) fn write_compacted(&self, dest: &Path, keep: &[u64]) -> Result<()> {
        self.file.write_compacted(dest, keep)
    }

    /// Discard vectors from `count` on (uncommitted after a crash).
    pub(crate) fn truncate_to(&mut self, count: u64) -> Result<()> {
        self.file.truncate_to(count)
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.file.count()
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Zero-copy access to one vector
    #[inline]
    pub fn get(&self, index: u64) -> Option<&[f32]> {
        bytemuck::try_cast_slice(self.file.get(index)?).ok()
    }

    /// Zero-copy access to `count` consecutive vectors
    pub fn get_batch(&self, start: u64, count: usize) -> Option<Vec<&[f32]>> {
        let bytes = self.file.range(start, count)?;
        let floats: &[f32] = bytemuck::try_cast_slice(bytes).ok()?;
        Some(floats.chunks_exact(self.dimension).collect())
    }

    /// Bytes used by stored vectors
    pub fn memory_usage(&self) -> usize {
        super::header::VectorHeader::SIZE + self.count() as usize * self.file.record_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn vectors_round_trip_and_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let mut storage = VectorStorage::new(path, 4).unwrap();
            assert_eq!(storage.append(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 0);
            let ids = storage.append_batch(&[vec![5.0; 4], vec![6.0; 4]]).unwrap();
            assert_eq!(ids, vec![1, 2]);
            assert_eq!(storage.get(0).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(
                storage.get_batch(1, 2).unwrap(),
                vec![&[5.0; 4][..], &[6.0; 4][..]]
            );
            storage.flush().unwrap();
        }
        let storage = VectorStorage::new(path, 4).unwrap();
        assert_eq!(storage.count(), 3);
        assert_eq!(storage.get(2).unwrap(), &[6.0; 4]);
    }

    #[test]
    fn rejects_wrong_dimension() {
        let dir = TempDir::new().unwrap();
        let mut storage = VectorStorage::new(dir.path().to_str().unwrap(), 4).unwrap();
        assert!(storage.append(&[1.0; 3]).is_err());
        assert!(storage.append_batch(&[vec![1.0; 4], vec![1.0; 5]]).is_err());
        assert_eq!(storage.count(), 0);
    }
}
