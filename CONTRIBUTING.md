# Contributing to SrvDB

Thanks for helping. SrvDB is an embedded vector database whose first job
is keeping data safe, so changes are judged on correctness first, then
clarity, then speed.

The full engineering standards are in [AGENTS.md](AGENTS.md). They apply
to human and AI contributors alike. This file covers the workflow.

## Set up

Requirements: stable Rust (`rustup update`), Python 3.9+, and
[uv](https://docs.astral.sh/uv/) or pip.

```bash
git clone https://github.com/Srinivas26k/srvdb
cd srvdb
cargo test --workspace            # Rust core tests

uv venv && source .venv/bin/activate
uv pip install maturin pytest
maturin develop                   # builds python/srvdb/_low_level
pytest tests/python               # Python tests
```

## Where things live

| Path | Contents |
| --- | --- |
| `crates/srvdb/` | The database engine (pure Rust) |
| `crates/srvdb-py/` | Python bindings (PyO3), conversion and error mapping only |
| `python/srvdb/` | Python package and `_low_level.pyi` type stubs |
| `docs/adr/` | Architecture decision records |
| `docs/engineering-plan.md` | Roadmap and what is being worked on |

Start with `crates/srvdb/src/lib.rs` for the public API and
`crates/srvdb/src/engine/` for how a call flows through the system.

## Making a change

1. Open or pick an issue, and say you are working on it.
2. Create a branch from `master`: `fix/<topic>`, `feat/<topic>`,
   `docs/<topic>`.
3. Write a failing test first for bug fixes.
4. Keep each commit to one logical change and stage files explicitly.
5. Before pushing, run:

   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```

6. Open a pull request describing what changed, why, and how you tested
   it.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```text
fix(storage): truncate orphan bytes past the committed count on open

After a crash, vectors.bin could contain bytes beyond the count in the
header. Reopening appended after them, so new ids pointed at the wrong
offsets. Truncate to the committed length before accepting writes.
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`,
`chore`. The body explains the why for a reviewer without context.

## What reviewers check

- Tests: a regression test for fixes; reopen tests for storage changes;
  recall measured against exact search for index changes.
- No source file over 300 lines; no new `unwrap()` on user or disk input.
- Errors are returned, never swallowed; no `println!` in library code.
- Public API changes are documented, reflected in the `.pyi` stubs, and
  noted in `CHANGELOG.md`.
- Performance claims come with a reproducible benchmark (hardware, OS,
  dataset, command).
- On-disk format changes include a version bump, a reader for the old
  format, and an ADR.

## Larger changes

For new index types, file-format changes, new dependencies or public API
redesigns, open an issue first. Significant decisions are recorded as an
ADR (copy `docs/adr/0000-template.md`).

## Reporting bugs

Include: OS and CPU (SIMD paths differ by architecture), SrvDB version,
Python version, the mode used, and a minimal script that reproduces the
problem. For data corruption, say whether the process was killed or the
machine lost power.

## Security issues

Do not open a public issue for a vulnerability. Email
<srinivasvarma764@gmail.com> with details and a reproduction.

## Licence

SrvDB is licensed under AGPL-3.0. By contributing you agree that your
contribution is licensed under the same terms. Because the project also
offers commercial licences, maintainers may ask significant contributors
to sign a contributor licence agreement before merging.
