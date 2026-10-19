//! On-disk header shared by the vector storage files

/// Internal vector file header (now with dimension info)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct VectorHeader {
    pub magic: u32,
    pub version: u16,
    pub dimension: u16, // NEW: Support 128-4096
    pub count: u64,
    pub quantization_mode: u8, // 0=None, 1=Scalar, 2=Product
    pub index_type: u8,        // 0=Flat, 1=HNSW, etc.
    pub reserved: [u8; 6],
}

impl VectorHeader {
    pub const MAGIC: u32 = 0x53764442; // "SrvDB"
    pub const VERSION: u16 = 4; // Version 4: Dynamic dimensions
    pub const SIZE: usize = std::mem::size_of::<VectorHeader>();

    pub fn new(dimension: usize) -> anyhow::Result<Self> {
        if dimension > u16::MAX as usize {
            anyhow::bail!("Dimension {} exceeds maximum {}", dimension, u16::MAX);
        }

        Ok(Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            dimension: dimension as u16,
            count: 0,
            quantization_mode: 0,
            index_type: 0,
            reserved: [0; 6],
        })
    }

    pub fn new_quantized(dimension: usize, mode: u8) -> anyhow::Result<Self> {
        if dimension > u16::MAX as usize {
            anyhow::bail!("Dimension {} exceeds maximum {}", dimension, u16::MAX);
        }

        Ok(Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            dimension: dimension as u16,
            count: 0,
            quantization_mode: mode,
            index_type: 0,
            reserved: [0; 6],
        })
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.magic != Self::MAGIC {
            anyhow::bail!(
                "Invalid magic number: expected 0x{:08X}, got 0x{:08X}",
                Self::MAGIC,
                self.magic
            );
        }

        if self.version > Self::VERSION {
            anyhow::bail!("Unsupported version: {}", self.version);
        }

        if !(128..=4096).contains(&(self.dimension as usize)) {
            anyhow::bail!("Invalid dimension: {}", self.dimension);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_validation() {
        let header = VectorHeader::new(1536).unwrap();
        assert!(header.validate().is_ok());

        // Test invalid dimension
        let invalid_header = VectorHeader::new(50).unwrap();
        assert!(invalid_header.validate().is_err());
    }
}
