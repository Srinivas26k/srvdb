//! AVX2 kernel: `vpshufb` lookups of 32 codes per instruction
//!
//! The u8 results are added as u16 pairs: `a0` gets each result as a u16
//! (even byte + 256 * odd byte), `a1` gets it shifted right by 8 (the odd
//! byte). After a chunk, odd sums are `a1` and even sums `a0 - (a1 << 8)`
//! (exact modulo 2^16, and each true sum is below 2^16).
#![cfg(target_arch = "x86_64")]

use super::{BLOCK, CHUNK};
use std::arch::x86_64::*;

/// # Safety
/// The CPU must support AVX2; `block` must hold `code_bytes * 32` bytes
/// and `luts` `code_bytes * 32` bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn block_sums(block: &[u8], luts: &[u8], code_bytes: usize) -> [u32; BLOCK] {
    let mut acc = [0u32; BLOCK];
    let mask = _mm256_set1_epi8(0x0F);
    let mut start = 0;
    while start < code_bytes {
        let end = (start + CHUNK).min(code_bytes);
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        for j in start..end {
            // SAFETY: j < code_bytes, so the 32 bytes at j * 32 lie inside
            // both slices (caller contract); loadu has no alignment need
            let (x, tl, th) = unsafe {
                (
                    _mm256_loadu_si256(block.as_ptr().add(j * BLOCK) as *const __m256i),
                    _mm_loadu_si128(luts.as_ptr().add(j * 32) as *const __m128i),
                    _mm_loadu_si128(luts.as_ptr().add(j * 32 + 16) as *const __m128i),
                )
            };
            let lo = _mm256_and_si256(x, mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask);
            let r1 = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(tl), lo);
            let r2 = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(th), hi);
            a0 = _mm256_add_epi16(a0, _mm256_add_epi16(r1, r2));
            a1 = _mm256_add_epi16(
                a1,
                _mm256_add_epi16(_mm256_srli_epi16(r1, 8), _mm256_srli_epi16(r2, 8)),
            );
        }
        let even = _mm256_sub_epi16(a0, _mm256_slli_epi16(a1, 8));
        let (mut e, mut o) = ([0u16; 16], [0u16; 16]);
        // SAFETY: each array is 32 bytes; storeu has no alignment need
        unsafe {
            _mm256_storeu_si256(e.as_mut_ptr() as *mut __m256i, even);
            _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, a1);
        }
        // u16 lane i covers bytes 2i (vector 2i) and 2i + 1
        for i in 0..16 {
            acc[2 * i] += e[i] as u32;
            acc[2 * i + 1] += o[i] as u32;
        }
        start = end;
    }
    acc
}
