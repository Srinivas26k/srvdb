//! Readers for v0.2 quantizer files (bincode 1.x, little-endian, fixed
//! 8-byte lengths), so databases created with v0.2 keep opening after
//! bincode was removed. Only these two files need a reader: v0.2 index
//! files are derived data and are rebuilt instead.

use crate::error::Error;
use crate::quant::codebook::Codebook;
use crate::quant::product::ProductQuantizer;
use crate::quant::scalar::ScalarQuantizer;
use anyhow::Result;

/// Sequential little-endian reader over a byte slice
struct Reader<'a> {
    bytes: &'a [u8],
    file: &'static str,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.bytes.len() < n {
            return Err(Error::Corrupt(format!("{} (v0.2) is truncated", self.file)).into());
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    /// A length prefix, rejected if it exceeds the bytes left (so a damaged
    /// file cannot trigger a huge allocation)
    fn len(&mut self, item_size: usize) -> Result<usize> {
        let n = self.u64()? as usize;
        if n.saturating_mul(item_size) > self.bytes.len() {
            return Err(
                Error::Corrupt(format!("{} (v0.2) has an invalid length", self.file)).into(),
            );
        }
        Ok(n)
    }

    fn f32_vec(&mut self) -> Result<Vec<f32>> {
        let n = self.len(4)?;
        let raw = self.take(n * 4)?;
        let (chunks, _) = raw.as_chunks::<4>(); // `take` returned exactly n * 4 bytes
        Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    fn finish<T>(self, value: T) -> Result<T> {
        if !self.bytes.is_empty() {
            return Err(Error::Corrupt(format!("{} (v0.2) has trailing bytes", self.file)).into());
        }
        Ok(value)
    }
}

/// `scalar_quantizer.bin` as written by v0.2
pub(crate) fn scalar_quantizer(bytes: &[u8]) -> Result<ScalarQuantizer> {
    let mut r = Reader {
        bytes,
        file: "scalar_quantizer.bin",
    };
    let min_vals = r.f32_vec()?;
    let max_vals = r.f32_vec()?;
    let dimension = r.u64()? as usize;
    r.finish(ScalarQuantizer {
        min_vals,
        max_vals,
        dimension,
    })
}

/// `codebooks.bin` as written by v0.2
pub(crate) fn product_quantizer(bytes: &[u8]) -> Result<ProductQuantizer> {
    let mut r = Reader {
        bytes,
        file: "codebooks.bin",
    };
    let m = r.u64()? as usize;
    let k = r.u64()? as usize;
    let d_sub = r.u64()? as usize;
    let codebooks = (0..r.len(8)?)
        .map(|_| {
            let centroids = (0..r.len(8)?).map(|_| r.f32_vec()).collect::<Result<_>>()?;
            Ok(Codebook { centroids })
        })
        .collect::<Result<_>>()?;
    r.finish(ProductQuantizer {
        m,
        k,
        d_sub,
        codebooks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQ: &[u8] = include_bytes!("../../tests/data/v0.2/scalar_quantizer.bin");
    const PQ: &[u8] = include_bytes!("../../tests/data/v0.2/codebooks.bin");

    #[test]
    fn reads_v02_scalar_quantizer() {
        let sq = scalar_quantizer(SQ).unwrap();
        assert_eq!(sq.min_vals, vec![-1.0, -0.5, 0.0, 0.25]);
        assert_eq!(sq.max_vals, vec![1.0, 0.5, 2.0, 0.75]);
        assert_eq!(sq.dimension, 4);
    }

    #[test]
    fn reads_v02_codebooks() {
        let pq = product_quantizer(PQ).unwrap();
        assert_eq!((pq.m, pq.k, pq.d_sub), (2, 3, 2));
        assert_eq!(pq.codebooks[0].centroids[1], vec![1.0, 0.0]);
        assert_eq!(pq.codebooks[1].centroids[2], vec![0.25, 0.75]);
    }

    #[test]
    fn damaged_files_are_corrupt() {
        for bad in [&SQ[..SQ.len() - 1], &[0xFF; 16][..]] {
            let err = scalar_quantizer(bad).unwrap_err();
            assert!(matches!(Error::from(err), Error::Corrupt(_)));
        }
        let err = product_quantizer(&PQ[..20]).unwrap_err();
        assert!(matches!(Error::from(err), Error::Corrupt(_)));
    }
}
