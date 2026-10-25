# AGENTS.md

Rules for AI coding agents (Claude Code, Codex, Cursor, Copilot and others)
working in this repository. Human contributors follow the same standards;
see [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

These rules are mandatory. When a rule conflicts with a request, say so
before acting instead of silently breaking the rule.

## 1. What SrvDB is

An embedded, single-node vector database for local and offline AI
applications. It runs inside the host process: there is no server, no
network access and no background service. Its first duty is to keep
committed data safe; speed and features come after that.

## 2. Repository layout

```
crates/srvdb/        Pure Rust core. No PyO3, no Python types.
  src/lib.rs           Module declarations and public re-exports only
  src/engine/          SrvDB: open, constructors, write, read, commit, modes
  src/storage/         Vector files, header, metadata (redb), index files
  src/index/           HNSW graph, IVF partitions, hybrid search
  src/quant/           Scalar (SQ8) and product (PQ) quantization
  src/distance/        Similarity kernels and scans
  src/tuning/          Automatic mode selection
  tests/ benches/ examples/
crates/srvdb-py/     PyO3 glue only: argument conversion, error mapping
python/srvdb/        Python package (src layout); _low_level.pyi stubs
docs/                ADRs (docs/adr/), release notes, plans, reports
website/             Documentation site (Astro Starlight, deployed to GitHub Pages)
```

Dependency direction is one way: `srvdb-py -> srvdb`. The core crate must
never depend on PyO3 or know that Python exists.

## 3. Commands

```bash
cargo build --workspace --all-targets
cargo test --workspace                 # all Rust tests
cargo fmt --all                        # format
cargo clippy --workspace --all-targets -- -D warnings
maturin develop                        # build the Python package into the active venv
pytest tests/python                    # Python tests, against the built extension
```

Run tests and clippy before every commit. Never report work as done
without running them; quote failures verbatim.

## 4. Architecture rules

- **Single responsibility.** One module, one concern. Split by what the
  code does (open, write, read, commit), not by type.
- **300-line limit.** No source file over 300 lines, tests included. Split
  along a real seam; never cram code to fit.
- **Bindings stay thin.** `crates/srvdb-py` converts arguments, calls the
  core, and maps errors to Python exceptions. Validation and logic that
  matter to Rust users belong in the core.
- **Public API is deliberate.** Everything exported from `lib.rs` is a
  compatibility promise. Prefer `pub(crate)`; add `#[non_exhaustive]` to
  new public config structs and enums.
- **Indexes are derived data.** HNSW and IVF must always be rebuildable
  from vector storage. Never make an index the only copy of anything.
- **No premature optimization.** Add complexity only with a benchmark that
  shows the need. No new index types or frameworks without an ADR.

## 5. Correctness and durability

- Define the guarantee before writing storage code: what is durable after
  `persist()` returns, and what may be lost after a crash. Never weaken a
  documented guarantee silently.
- Writes to an existing file that must survive a crash use
  write-to-temp, `fsync`, atomic rename.
- On open, validate every on-disk structure (magic, version, sizes) and
  fail with a clear error instead of reading garbage.
- Any change to an on-disk format bumps its version, keeps a reader for
  the previous version, and adds a migration test.
- `unsafe` requires a `// SAFETY:` comment explaining why it is sound.
  Prefer safe alternatives (`bytemuck`, explicit little-endian encoding).

## 6. Errors

- Never swallow errors: no `let _ = fallible()`, no `.ok()` to discard, no
  empty `Err(_) => {}` arms. If ignoring is truly correct, comment why.
- No `unwrap()`/`expect()` on paths reachable from user input or disk
  contents. They are allowed in tests and for proven invariants with a
  comment.
- Return typed, domain-specific errors from the public API. Messages say
  what went wrong and how to fix it; they never include vector data or
  metadata contents.
- Python maps errors to specific exception types (`ValueError` for bad
  input, `OSError` for I/O, `RuntimeError` for internal faults).

## 7. Input validation

Validate at the public boundary (core API and bindings), with allow-lists:

- vector dimension matches the database; all values finite (no NaN/inf)
- `k >= 1`; batch lengths match; IDs non-empty and unique
- metadata is valid JSON; mode and parameter names from a fixed set
- paths are used as given; never build paths from metadata content

## 8. Logging

- Library code never writes to stdout/stderr (`println!`, `eprintln!`).
  Use the `log` crate; the host application chooses format (JSON or not)
  and destination.
- Never log vector values, metadata contents or user identifiers.
  Log counts, sizes, durations, file names and error kinds.

## 9. Security and supply chain

- No secrets, tokens or credentials in code, tests, fixtures or history.
- No network access from the library. No code execution from data.
- Nothing requires root; create files with default permissions.
- Commit `Cargo.lock` and `uv.lock`. New dependencies need a reason in the
  commit message, a compatible licence (MIT, Apache-2.0, BSD, ISC,
  Unlicense, Zlib) and an active maintainer. CI runs `cargo deny` (licences
  and advisories) and generates an SBOM for releases.
- Use established, reviewed libraries for hashing and checksums; never
  hand-roll cryptography.

## 10. Testing

- Every bug fix ships with a test that fails before the fix.
- Every storage change ships with a reopen test; durability changes ship
  with a crash test (kill the process mid-write, reopen, verify).
- Approximate search is tested against exact search (recall), not
  against hard-coded IDs.
- Python tests live in `tests/python/` and run against the built wheel.
- Keep debug-mode test time reasonable; mark slow tests `#[ignore]` and run
  them in CI's release job.

## 11. Benchmarks and claims

Never state a performance number that is not produced by a committed,
reproducible benchmark. Every published number lists hardware, OS,
dataset, commit hash and the command used.

## 12. Documentation

- Every public item has a doc comment saying what it does and its
  errors; mention durability and thread-safety where relevant.
- Keep `python/srvdb/_low_level.pyi` in sync with the bindings in the same
  commit that changes them.
- Significant decisions (format changes, new dependencies, new index
  types, public API changes) get an ADR in `docs/adr/` using the template.
- Update `CHANGELOG.md` under "Unreleased" for user-visible changes.
- User-facing behaviour changes update the matching page in
  `website/src/content/docs/` in the same branch. Every code snippet there
  must run against the current code; `npm run build` in `website/` must pass.

## 13. Commits and branches

- Branch model:
  - `master` holds released code only. It changes only by merging a
    finished `release/X.Y` branch.
  - `release/X.Y` (e.g. `release/0.3`) integrates one version. Never
    commit to it directly; merge work branches into it.
  - Work branches start from the current `release/X.Y` and are named
    for one tracker track or task: `feat/<topic>`, `fix/<topic>`,
    `docs/<topic>`, `ci/<topic>`, `refactor/<topic>`.
  - Merge a work branch into `release/X.Y` with `git merge --no-ff`
    only when its tests, clippy and pre-commit hooks pass. The merge
    commit summarizes what the branch delivered.
- One logical change per commit. Stage specific files (`git add <path>`),
  never `git add -A` or `git add .`.
- Conventional Commits: `type(scope): summary` with types `feat`, `fix`,
  `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.
- The body explains what changed and why, in enough detail for a
  reviewer who has not seen the conversation.
- Do not add AI attribution lines (such as `Co-Authored-By` for an AI
  tool) to commit messages.
- Never push, tag, publish or bump versions unless explicitly asked.
  Pushing a `v*` tag publishes to PyPI.

## 14. What not to do

- Do not add distributed features, GPU code, servers or dashboards.
- Do not change the licence or licence metadata.
- Do not reformat or rename unrelated code in a feature commit.
- Do not delete or weaken a failing test to make CI pass.
- Do not commit generated files (`target/`, built `.so`/`.pyd`, benchmark
  databases).

## 15. Standards that do not apply here, and why

Some general service-design guidelines do not fit an embedded database:

- Statelessness and horizontal scaling: SrvDB *is* the state; scaling is
  the host application's concern.
- Load balancers, rate limiting, backpressure: no network or request queue.
  Large batch inputs are still bounded and validated.
- SQL injection and XSS: SrvDB builds no SQL and renders no HTML; `redb`
  takes typed keys.
- Correlation IDs: meaningful only across services; the host can attach
  its own context to `log` records.
