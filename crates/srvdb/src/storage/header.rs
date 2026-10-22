//! On-disk header shared by the vector storage files
//!
//! Layout (24 bytes, little-endian), unchanged since v0.2 so existing files
//! open as before:
//!
//! | offset | size | field             |
//! |--------|------|-------------------|
//! | 0      | 4    | magic             |
//! | 4      | 2    | version           |
//! | 6      | 2    | dimension         |
//! | 8      | 8    | count             |
//! | 16     | 1    | quantization mode |
//! | 17     | 1    | index type        |
//! | 18     | 6    | reserved (zero)   |
//!
//! Since v0.3 the authoritative committed count lives in `redb`
//! (ADR 0003); the header count is kept up to date for tools and for
//! opening v0.2 databases.

use crate::error::Error;

/// Header at the start of every vector storage file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorHeader {
    pub magic: u32,
    pub version: u16,
    pub dimension: u16,
    pub count: u64,
    pub quantization_mode: u8, // 0 = none, 1 = scalar (SQ8), 2 = product (PQ)
    pub index_type: u8,
    pub reserved: [u8; 6],
}

impl VectorHeader {
    pub const MAGIC: u32 = 0x5376_4442; // "SvDB" in little-endian order
    pub const VERSION: u16 = 4;
    pub const SIZE: usize = 24;

    pub fn new(dimension: usize) -> anyhow::Result<Self> {
        Self::new_quantized(dimension, 0)
    }

    pub fn new_quantized(dimension: usize, mode: u8) -> anyhow::Result<Self> {
        let dimension = u16::try_from(dimension).map_err(|_| {
            Error::invalid_input(format!("dimension {dimension} exceeds {}", u16::MAX))
        })?;
        Ok(Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            dimension,
            count: 0,
            quantization_mode: mode,
            index_type: 0,
            reserved: [0; 6],
        })
    }

    /// Encode as the 24-byte on-disk form
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..4].copy_from_slice(&self.magic.to_le_bytes());
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&self.dimension.to_le_bytes());
        out[8..16].copy_from_slice(&self.count.to_le_bytes());
        out[16] = self.quantization_mode;
        out[17] = self.index_type;
        out[18..24].copy_from_slice(&self.reserved);
        out
    }

    /// Decode and validate a header read from disk
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(Error::Corrupt(format!(
                "file is {} bytes, shorter than the {}-byte header",
                bytes.len(),
                Self::SIZE
            ))
            .into());
        }
        let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let mut count = [0u8; 8];
        count.copy_from_slice(&bytes[8..16]);
        let mut reserved = [0u8; 6];
        reserved.copy_from_slice(&bytes[18..24]);
        let header = Self {
            magic: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            version: u16_at(4),
            dimension: u16_at(6),
            count: u64::from_le_bytes(count),
            quantization_mode: bytes[16],
            index_type: bytes[17],
            reserved,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.magic != Self::MAGIC {
            return Err(Error::Corrupt(format!(
                "not a SrvDB vector file (magic 0x{:08X})",
                self.magic
            ))
            .into());
        }
        if self.version > Self::VERSION {
            return Err(Error::Corrupt(format!(
                "vector file version {} is newer than this SrvDB supports ({})",
                self.version,
                Self::VERSION
            ))
            .into());
        }
        if !(1..=4096).contains(&(self.dimension as usize)) {
            return Err(Error::Corrupt(format!("invalid dimension {}", self.dimension)).into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_bytes() {
        let mut header = VectorHeader::new_quantized(768, 2).unwrap();
        header.count = 123_456;
        let bytes = header.to_bytes();
        assert_eq!(VectorHeader::from_bytes(&bytes).unwrap(), header);
    }

    #[test]
    fn matches_v02_layout() {
        // Bytes as v0.2 wrote them (repr(C) struct on a little-endian CPU)
        let mut v02 = Vec::new();
        v02.extend_from_slice(&0x5376_4442u32.to_le_bytes());
        v02.extend_from_slice(&4u16.to_le_bytes());
        v02.extend_from_slice(&1536u16.to_le_bytes());
        v02.extend_from_slice(&42u64.to_le_bytes());
        v02.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        let header = VectorHeader::from_bytes(&v02).unwrap();
        assert_eq!(
            (header.dimension, header.count, header.quantization_mode),
            (1536, 42, 1)
        );
    }

    #[test]
    fn rejects_foreign_or_truncated_files() {
        let err = VectorHeader::from_bytes(&[0u8; 24]).unwrap_err();
        assert!(matches!(Error::from(err), Error::Corrupt(_)));
        let err = VectorHeader::from_bytes(&[0u8; 10]).unwrap_err();
        assert!(matches!(Error::from(err), Error::Corrupt(_)));
    }
}
