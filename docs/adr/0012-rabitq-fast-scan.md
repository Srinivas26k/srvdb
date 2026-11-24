# ADR 0012: Fast-scan kernel for RaBitQ codes

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

ADR 0008 scored each RaBitQ record with one 256-entry f32 table per code
byte. Every code byte costs a data-dependent gather from a table that
does not fit in registers. The scan was therefore gather-bound, and at
dimension 768 the 2-bit and 4-bit modes were barely faster than flat
(issue #24; numbers in `docs/benchmarks.md`). Each query also built those
tables (`code_bytes * 256` entries) and rotated the query with a dense
`D x D` product summed as one serial f32 chain.

## Decision

1. **Nibble tables.** A query gets one 16-entry f32 table per nibble of a
   code byte. A nibble holds `4 / b` dimensions (one dimension at 4 bits,
   two at 2 bits, four at 1 bit), so every bit width shares one kernel
   and the record bytes are read as they are today.
2. **u8 tables.** Each table is shifted by its minimum and divided by one
   step shared by all tables (`step = max range / 255`), then rounded to
   u8. The dot product is `step * sum + sum of minimums`. Rounding adds at
   most `M * step / 2` absolute error to the dot product for `M` tables,
   scaled by the record's correction factor. Rerank removes this error
   from the returned scores. With rerank off, results are estimates, as
   they always were.
3. **Block-interleaved in-memory copy.** Codes are copied into blocks of
   32 vectors, with byte `j` of every vector side by side
   (`block[j * 32 + v]`), and the f32 factors in a separate array. The
   copy is built from `rabitq_codes.bin` on open and kept in step on
   append and truncate. **The file format does not change.** No version
   bump or migration is needed, and the copy is derived data like the file.
4. **Kernels.** `Kernel::detect()` picks AVX2 (`is_x86_feature_detected!`)
   on x86_64, NEON on aarch64, and otherwise the scalar kernel. The scalar
   kernel is always compiled and is the test reference.
   - AVX2: load 32 code bytes, split the nibbles, and look up 32 values
     per `vpshufb` from the 16-byte table broadcast to both lanes. The u8
     results are added as u16 pairs: one accumulator takes the raw u16
     (even + 256 * odd), the other takes it shifted right by 8 (odd).
     Even sums are recovered as `a0 - (a1 << 8)`, which is exact modulo
     2^16.
   - NEON: two halves of 16 vectors, `vqtbl1q_u8` lookups, widened to
     u16 with `vaddw_u8` / `vaddw_high_u8`.
   - Both widen to u32 every 64 code bytes (at most 2 * 255 * 64 = 32640
     per u16 lane), so the integer sums are **identical** to the scalar
     ones for any input. A property test checks this, including all-255
     tables and lengths that are not a multiple of 64.
5. **Parallel scan.** Rayon tasks of 32 blocks (1024 vectors) each keep
   their own top `k * rerank`. The deleted-id check is skipped when
   nothing is deleted.
6. **Rotation.** The dense rotation stays (no Hadamard rotation; issue #53
   is still open). Its row products now sum in eight independent lanes,
   which the compiler vectorizes. The matrix and its fingerprint do not
   change, so existing codes stay valid. Codes encoded after this change
   can differ from older ones only by f32 rounding.

## Alternatives considered

- **Version-2 interleaved code file.** It would avoid the open-time copy,
  but it needs a new append path for partial blocks, a migration and a
  reader for version 1. Building the copy takes one pass over the codes,
  small next to the O(n D^2) work of encoding them, so we kept the file.
- **u16 or f32 tables in registers.** They allow fewer lookups per
  instruction or need gathers again. u8 is what FAISS fast-scan and
  RaBitQ use.
- **Hadamard rotation (#53).** It would cut the O(D^2) cost per query,
  but the lane-summed dense product was enough for the target. It stays
  a separate decision because of the padding it needs.
- **Saturating u8 adds.** They lose exactness and the scalar/SIMD
  identity.

## Consequences

- Resident memory for a RaBitQ database grows by one copy of the codes
  (`n * ceil(D * b / 8)` bytes, plus up to 31 vectors of padding). The
  mapped code file is no longer read by queries.
- Opening a database pays one extra pass over the codes.
- Each new SIMD kernel must match the scalar sums exactly, and the
  property test enforces this. The NEON kernel is type-checked by
  `cargo check --target aarch64-unknown-linux-gnu`, but it is only run on
  aarch64 CI or hardware.
