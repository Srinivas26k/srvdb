# ADR 0002: Typed errors at the public boundary

- **Status:** Accepted
- **Date:** 2026-10-21

## Context

The public API returned `anyhow::Error`. Rust callers could only inspect
message strings, and the Python bindings mapped every failure to
`RuntimeError`, so applications could not tell a bad argument from a
disk failure or a bug.

## Decision

- The public API returns `srvdb::Result<T>` with a `#[non_exhaustive]`
  `srvdb::Error` enum: `InvalidInput`, `DimensionMismatch`,
  `InvalidState`, `Corrupt`, `Io`, `Internal`.
- Internal modules keep `anyhow` for context chains. A single
  `From<anyhow::Error>` at the boundary recovers typed errors raised
  inside, classifies I/O errors and maps the rest to `Internal`.
- Python maps `InvalidInput`, `DimensionMismatch` and `InvalidState` to
  `ValueError`, `Io` to `OSError`, and `Corrupt` and `Internal` to
  `RuntimeError`.
- Messages never include vector values or metadata contents.

## Alternatives considered

- **Typed errors in every module:** most precise, but a large rewrite of
  storage code that will change again in the durability work.
- **Keep anyhow publicly:** no way for callers to branch on the cause.

## Consequences

- Callers can match on the error kind; `#[non_exhaustive]` lets variants
  be added without a breaking change.
- Internal code that should produce a specific kind must raise
  `srvdb::Error` (wrapped in `anyhow` is fine); anything else surfaces
  as `Internal`, so misclassification shows up as a visible bug.
