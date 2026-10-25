# ADR 0006: Relicense SrvDB under Apache-2.0

- **Status:** Accepted
- **Date:** 2026-10-25

## Context

SrvDB 0.1 and 0.2 were published under AGPL-3.0, with commercial licences
offered for proprietary use. The metadata also disagreed: `Cargo.toml` said
`AGPL-3.0-only`, the PyPI classifier said AGPLv3-or-later.

SrvDB's goal is to become the default embedded vector database for local
AI applications, used by individual developers and large companies alike.
SrvDB is a library linked into the host application, so a copyleft licence
applies to the host. Many companies forbid AGPL code outright, which stops
engineers there from even evaluating it. Comparable projects are
permissive: FAISS and DuckDB (MIT); Chroma, Qdrant, LanceDB and Milvus
(Apache-2.0).

Every commit to date is by the maintainer, so the maintainer holds the
full copyright and can relicense without anyone else's permission.

## Decision

From 0.3.0, SrvDB (both crates and the Python package) is licensed under
the Apache License 2.0, recorded as the SPDX expression `Apache-2.0` in
`Cargo.toml` and `pyproject.toml`, with a `NOTICE` file. Releases up to and
including 0.2.0 remain available under AGPL-3.0, as published.

Revenue comes from support, consulting and, later, separately licensed
products or services built around the open core, not from licence
restrictions on the core.

## Alternatives considered

- **Keep AGPL-3.0 with commercial licences:** protects against a cloud
  provider reselling SrvDB, but an embedded library has little to resell
  as a service, and the licence blocks the adoption the project needs.
- **MIT:** equally permissive and simpler, but has no explicit patent
  grant, which corporate legal reviews prefer.
- **Dual MIT OR Apache-2.0 (Rust convention):** more choice for users, but
  two licences to explain for no gain over Apache-2.0 alone for a Python-
  first audience.
- **Source-available (BSL, SSPL, Elastic):** not open source under the OSI
  definition; deters the same companies and contributors.

## Consequences

- Anyone, including competitors, may use, modify and ship SrvDB in closed
  products. The project's advantage must come from quality, speed of
  development and community, not the licence.
- Selling commercial licences to the core is no longer possible. Separate
  commercial products remain possible while the maintainer holds the
  copyright; outside contributions should therefore come under a CLA
  (CONTRIBUTING.md) so future licensing decisions stay open.
- `cargo deny` no longer needs an exception for SrvDB's own crates.
