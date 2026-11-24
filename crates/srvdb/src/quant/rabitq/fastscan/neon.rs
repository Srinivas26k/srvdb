//! NEON kernel: `tbl` lookups of 16 codes per instruction, two halves
//! per block, widened to u16 with `uaddw`
#![cfg(target_arch = "aarch64")]

use super::{BLOCK, CHUNK};
use std::arch::aarch64::*;

/// # Safety
/// The CPU must support NEON; `block` must hold `code_bytes * 32` bytes
/// and `luts` `code_bytes * 32` bytes.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn block_sums(block: &[u8], luts: &[u8], code_bytes: usize) -> [u32; BLOCK] {
    let mut acc = [0u32; BLOCK];
    let mask = vdupq_n_u8(0x0F);
    for half in 0..2 {
        let mut start = 0;
        while start < code_bytes {
            let end = (start + CHUNK).min(code_bytes);
            let mut low = vdupq_n_u16(0);
            let mut high = vdupq_n_u16(0);
            for j in start..end {
                // SAFETY: j < code_bytes, so these 16-byte loads lie inside
                // both slices (caller contract)
                let (x, tl, th) = unsafe {
                    (
                        vld1q_u8(block.as_ptr().add(j * BLOCK + half * 16)),
                        vld1q_u8(luts.as_ptr().add(j * 32)),
                        vld1q_u8(luts.as_ptr().add(j * 32 + 16)),
                    )
                };
                let r1 = vqtbl1q_u8(tl, vandq_u8(x, mask));
                let r2 = vqtbl1q_u8(th, vshrq_n_u8(x, 4));
                low = vaddw_u8(vaddw_u8(low, vget_low_u8(r1)), vget_low_u8(r2));
                high = vaddw_high_u8(vaddw_high_u8(high, r1), r2);
            }
            let (mut l, mut h) = ([0u16; 8], [0u16; 8]);
            // SAFETY: each array holds exactly 8 u16
            unsafe {
                vst1q_u16(l.as_mut_ptr(), low);
                vst1q_u16(h.as_mut_ptr(), high);
            }
            for i in 0..8 {
                acc[half * 16 + i] += l[i] as u32;
                acc[half * 16 + 8 + i] += h[i] as u32;
            }
            start = end;
        }
    }
    acc
}
