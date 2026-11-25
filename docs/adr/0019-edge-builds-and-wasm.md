# ADR 0019: Edge builds, abi3 wheels and WebAssembly (no-go for now)

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

Issue #87 asks for wheels that run on small devices (Raspberry Pi 64-bit,
Apple Silicon, Windows), a smaller binary option, and a decision on a
WebAssembly build of the core.

The release workflow already built wheels for linux x86_64 and aarch64
(manylinux), macOS arm64 and x86_64, and Windows x86_64, once per CPython
version 3.9 to 3.14 (30 wheels). CI already ran `cargo test -p srvdb` on
`ubuntu-24.04-arm`.

Local measurement (x86_64 Linux, AMD Ryzen 5 3550H, rustc 1.100.0,
`maturin build`, commit base c59feb1; sizes only, not performance):

| Build                                   | `.so` bytes | wheel bytes |
|-----------------------------------------|------------:|------------:|
| `--release` (opt-level 3, fat LTO)      | 3,686,440   | 1,789,249   |
| `--release --features abi3`             | 3,702,760   | 1,796,396   |
| `--profile release-small` (opt-level z) | 2,744,584   | 1,301,711   |

The default release profile already uses fat LTO, one codegen unit and
`strip`, so the only remaining size lever without code changes is
`opt-level = "z"` (about 26% smaller `.so`, 27% smaller wheel). Its speed
cost is not measured here; the default stays tuned for speed.

WebAssembly, `cargo check -p srvdb` with rustc 1.100.0:

- `wasm32-unknown-unknown`: fails in `getrandom` 0.4 (via `rand`), which
  needs the `wasm_js` feature and a JavaScript host for entropy.
- `wasm32-wasip1`: fails in `redb` 2.6 (the migration-only `redb2`
  dependency), which uses `#![feature(wasi_ext)]` and needs nightly.
  `redb` 4.x compiled.

Compilation stops at the dependencies, so these are the first blockers,
not all of them. Further blockers from the design:

- **mmap:** vector files are read through `memmap2`; neither wasm target
  has `mmap`. A read-into-memory backend would be needed.
- **Threads:** `rayon` is used in 15 core modules. `wasm32-unknown-unknown`
  has no threads; wasip1 has none either (wasip1-threads is experimental).
- **File system and durability:** `wasm32-unknown-unknown` has no file
  system. Under WASI, `fsync` and atomic rename depend on the host runtime,
  so the durability guarantee of ADR 0003 could not be stated for every
  host. In a browser it would need OPFS through JavaScript bindings.
- **SIMD kernels:** `simsimd` is C code; it would need a wasm C toolchain
  and the `simd128` paths, or a pure Rust fallback.

## Decision

1. Release wheels are built with PyO3's stable ABI (`abi3`, minimum
   Python 3.9) through a new `srvdb-py` feature `abi3`. One wheel per
   platform covers CPython 3.9 and later, including future versions. The
   full Python test suite passed against the abi3 wheel on 3.9 and 3.13.
   Publish gating in the release workflow is unchanged.
2. A `release-small` Cargo profile (inherits `release`, `opt-level = "z"`)
   is available for users who build from source for constrained devices.
   It keeps unwinding: PyO3 converts Rust panics to `PanicException`, and
   `panic = "abort"` would kill the host process instead. Release wheels
   keep the default speed-tuned profile.
3. CI builds and tests the linux aarch64 abi3 wheel on pull requests into
   `release/*` branches.
4. **WebAssembly: no-go for 0.4.** A go would require, each behind a
   cargo feature that leaves the native build unchanged:
   - a storage backend without `mmap` (read files into memory);
   - a single-threaded path that removes the `rayon` requirement;
   - removing or feature-gating `redb2` (drop 0.2 migration on wasm);
   - `getrandom` with `wasm_js` for the browser target, or a seeded RNG
     supplied by the caller;
   - a pure Rust (or `simd128`) distance fallback for `simsimd`;
   - a written durability guarantee per host (WASI runtime, browser
     OPFS), with crash tests on each;
   - a CI job that builds and tests the wasm target.
   That is a core refactor with its own ADR, not part of this change.

## Alternatives considered

- **Per-version wheels (status quo):** 30 wheels per release and a new
  build every CPython release. abi3 had no measured size cost worth
  keeping them (+0.4% wheel size).
- **`opt-level = "z"` for release wheels:** smaller downloads, but an
  unmeasured slowdown of search kernels; rejected until a benchmark shows
  the cost.
- **`opt-level = "s"`:** not measured; `z` is the size end of the range
  and a user can override the profile locally.
- **Ship a WASM build with an in-memory store only:** would not keep the
  durability promise that is SrvDB's first duty.

## Consequences

- Fewer release artifacts; users on new CPython versions get a working
  wheel without a SrvDB release.
- abi3 limits the bindings to the limited Python C API. Any new binding
  that needs a non-limited API must be gated or redesigned.
- One more CI job on PRs into release branches (arm64 runner minutes).
- The wasm blocker list must be revisited if browser or WASI demand
  appears; it is the starting checklist for that ADR.
