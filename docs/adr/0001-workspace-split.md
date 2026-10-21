# ADR 0001: Split into a core crate and a Python bindings crate

- **Status:** Accepted
- **Date:** 2026-10-19

## Context

Until v0.2 a single crate contained both the database engine and the PyO3
bindings, switched by a `python` feature that was on by default. This
caused three problems:

- The core could not be built or tested without a Python toolchain, and a
  build with the feature off failed to compile.
- Binding code and engine code were interleaved, so engine logic leaked
  into `api/python.rs` and the boundary was unclear.
- Files had grown to 900+ lines, mixing unrelated responsibilities.

## Decision

- A Cargo workspace with two crates:
  - `crates/srvdb`: the pure-Rust engine, with no PyO3 dependency.
  - `crates/srvdb-py`: PyO3 glue that builds the `srvdb._low_level`
    extension module.
- A src-layout Python package in `python/srvdb/` that re-exports the
  extension and ships `_low_level.pyi` type stubs.
- Source files stay under 300 lines, split by responsibility.

## Alternatives considered

- **Keep the feature flag:** still lets binding code reach into engine
  internals, and every core build has to think about PyO3.
- **Separate repositories:** too much release overhead for one maintainer;
  core and bindings change together.

## Consequences

- Rust users depend on `srvdb` without pulling in PyO3.
- The bindings can only use the core's public API, which forces that API
  to be complete.
- `srvdb::core::*` paths were removed; `srvdb::types::*` remains.
- The Python crate has no Rust test target (an extension-module cdylib
  cannot link a test binary); bindings are tested from Python.
