//! Fast-scan kernels: every compiled kernel gives the scalar sums, and the
//! u8 estimate stays within the documented error of the f32 estimate

use super::*;
use crate::quant::rabitq::{RabitqQuantizer, SUPPORTED_BITS};

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 30) as f32) - 1.0
}

fn vectors(n: usize, dim: usize, mut seed: u64) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng(&mut seed)).collect())
        .collect()
}

fn encoded(q: &RabitqQuantizer, data: &[Vec<f32>]) -> (Vec<u8>, FastScanCodes) {
    let mut records = Vec::new();
    for v in data {
        q.encode_into(v, &mut records);
    }
    let mut codes = FastScanCodes::with_widths(q.dim(), q.bits());
    codes.extend(&records, q.record_size()).unwrap();
    (records, codes)
}

/// Random bytes stress every nibble value and the u16 widening, including
/// code lengths that are not a multiple of the widening chunk
#[test]
fn simd_kernels_match_scalar_exactly() {
    let mut seed = 5u64;
    for code_bytes in [1usize, 3, 63, 64, 65, 200, 384, 700] {
        let block: Vec<u8> = (0..code_bytes * BLOCK)
            .map(|_| (rng(&mut seed) * 127.0 + 128.0) as u8)
            .collect();
        // Worst case for overflow: every table entry is 255
        for fill in [None, Some(255u8)] {
            let luts: Vec<u8> = (0..code_bytes * 32)
                .map(|_| fill.unwrap_or((rng(&mut seed) * 127.0 + 128.0) as u8))
                .collect();
            let want = scalar::block_sums(&block, &luts, code_bytes);
            for kernel in Kernel::available() {
                let got = kernel.block_sums(&block, &luts, code_bytes);
                assert_eq!(got, want, "{kernel:?}, {code_bytes} code bytes");
            }
        }
    }
}

#[test]
fn quantized_estimates_stay_within_bound() {
    for bits in SUPPORTED_BITS {
        for dim in [37usize, 128, 384] {
            let q = RabitqQuantizer::new(dim, bits, 3);
            let data = vectors(70, dim, 1);
            let (records, codes) = encoded(&q, &data);
            for query in vectors(4, dim, 9) {
                let exact = q.prepare(&query);
                let lut = QueryLut::new(exact.tables());
                for kernel in Kernel::available() {
                    let scores = codes.scores(&lut, kernel);
                    assert_eq!(scores.len(), data.len());
                    for (i, r) in records.chunks_exact(q.record_size()).enumerate() {
                        let factor = f32::from_le_bytes([r[0], r[1], r[2], r[3]]);
                        let bound = factor.abs() * lut.max_error() + 1e-5;
                        let diff = (scores[i] - exact.score(r)).abs();
                        assert!(diff <= bound, "{bits} bits dim {dim}: {diff} > {bound}");
                    }
                }
            }
        }
    }
}

#[test]
fn appends_fill_partial_blocks_and_truncate_drops() {
    let q = RabitqQuantizer::new(20, 4, 1);
    let data = vectors(75, 20, 2);
    let (records, whole) = encoded(&q, &data);
    let mut parts = FastScanCodes::new(q.code_bytes());
    let size = q.record_size();
    for chunk in records.chunks(size * 7) {
        parts.extend(chunk, size).unwrap();
    }
    let lut = QueryLut::new(q.prepare(&data[3]).tables());
    assert_eq!(parts.len(), 75);
    assert_eq!(
        parts.scores(&lut, Kernel::SCALAR),
        whole.scores(&lut, Kernel::SCALAR)
    );
    parts.truncate(40);
    assert_eq!(parts.len(), 40);
    assert_eq!(
        parts.scores(&lut, Kernel::SCALAR)[..],
        whole.scores(&lut, Kernel::SCALAR)[..40]
    );
}

#[test]
fn zero_query_scores_zero() {
    let q = RabitqQuantizer::new(16, 2, 1);
    let (_, codes) = encoded(&q, &vectors(5, 16, 3));
    let lut = QueryLut::new(q.prepare(&[0.0; 16]).tables());
    assert!(codes
        .scores(&lut, Kernel::detect())
        .iter()
        .all(|s| *s == 0.0));
}

#[test]
fn impossible_reservation_is_an_error_not_an_abort() {
    let mut codes = FastScanCodes::new(1 << 20);
    match codes.reserve(1 << 50) {
        Err(Error::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::OutOfMemory),
        other => panic!("expected OutOfMemory, got {other:?}"),
    }
    assert_eq!(codes.len(), 0);
}
