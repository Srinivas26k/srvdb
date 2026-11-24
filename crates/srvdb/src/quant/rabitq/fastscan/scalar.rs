//! Portable kernel: always compiled, the reference for the SIMD kernels

use super::BLOCK;

/// Integer sums of one block (see [`super::Kernel::block_sums`])
pub(crate) fn block_sums(block: &[u8], luts: &[u8], code_bytes: usize) -> [u32; BLOCK] {
    let mut acc = [0u32; BLOCK];
    for j in 0..code_bytes {
        let lo = &luts[j * 32..j * 32 + 16];
        let hi = &luts[j * 32 + 16..j * 32 + 32];
        let codes = &block[j * BLOCK..(j + 1) * BLOCK];
        for (a, &b) in acc.iter_mut().zip(codes) {
            *a += lo[(b & 15) as usize] as u32 + hi[(b >> 4) as usize] as u32;
        }
    }
    acc
}
