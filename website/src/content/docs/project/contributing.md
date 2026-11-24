---
title: Contributing
description: Make your first contribution to SrvDB in about 15 minutes.
---

Contributions of every size are welcome: bug reports, docs fixes, tests,
benchmarks on your hardware, and code. Human contributors and AI-assisted
work follow the same rules.

## Set up

You need stable Rust (`rustup update`), Python 3.9 or newer, and
[uv](https://docs.astral.sh/uv/) or pip.

```bash
git clone https://github.com/Srinivas26k/srvdb
cd srvdb
cargo test --workspace            # Rust core tests

uv venv && source .venv/bin/activate
uv pip install maturin pytest
maturin develop                   # builds the Python extension
pytest tests/python               # Python tests
```

If both test runs pass, you are ready.

## Where to start

- [Good first issues](https://github.com/Srinivas26k/srvdb/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22):
  small, well-scoped, with pointers to the files involved.
- [Help wanted](https://github.com/Srinivas26k/srvdb/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22):
  larger tasks where outside help is especially useful.

Comment on the issue to say you are taking it, so work is not duplicated.

## Areas by skill

| Skill | Where | Good tasks |
| --- | --- | --- |
| Rust core | `crates/srvdb/src/` (`engine/`, `storage/`, `index/`, `quant/`) | Error handling, validation, new tests |
| Python API | `crates/srvdb-py/` (bindings), `python/srvdb/` (package and `_low_level.pyi`) | Bindings, type stubs, Python tests |
| Docs and examples | `website/src/content/docs/`, `examples/`, `crates/srvdb/examples/` | Fixing errors, runnable examples |
| Benchmarks | `benchmarks/`, `crates/srvdb/examples/bench/` | Running on your hardware, reproducibility |
| Testing | `crates/srvdb/tests/`, `tests/python/` | Reopen tests, crash tests, recall tests |

Start with `crates/srvdb/src/lib.rs` for the public API.

## Branches and commits

- Branch from the current release branch (for example `release/0.4`), not
  from `master`. Name the branch `feat/<topic>`, `fix/<topic>`,
  `docs/<topic>`, `ci/<topic>` or `refactor/<topic>`.
- Open the pull request against that release branch. `master` only receives
  finished releases.
- One logical change per commit. Stage files by name (`git add <path>`),
  not `git add -A`.
- Use [Conventional Commits](https://www.conventionalcommits.org/):
  `type(scope): summary`, with types `feat`, `fix`, `perf`, `refactor`,
  `test`, `docs`, `build`, `ci`, `chore`. The body explains what changed and
  why.
- Bug fixes include a test that fails without the fix. Storage changes need
  a reopen test; durability changes need a crash test.

Before pushing:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
pytest tests/python
```

Changes to the Python API update its page under the
[Python API reference](/srvdb/reference/python/). Docs site changes must pass
`npm run build` in `website/`.

## Proposing an idea

- Small ideas: open a
  [feature request](https://github.com/Srinivas26k/srvdb/issues/new/choose)
  describing the problem, not only the solution.
- Significant decisions (on-disk format changes, new dependencies, new index
  types, public API changes): open an issue first, then write an ADR by
  copying `docs/adr/0000-template.md`.

## How decisions are made

- **ADRs.** Significant decisions are written down as architecture decision
  records in `docs/adr/`. An accepted ADR is never rewritten; a later ADR
  supersedes it.
- **Maintainers.** SrvDB currently has one maintainer, who reviews and
  merges pull requests and decides releases.
- **Becoming a maintainer.** Contributors who make several careful,
  well-tested changes and help review others' work can be invited to become
  maintainers, with merge rights for the areas they know.

## Rules and conduct

The full engineering rules are in
[AGENTS.md](https://github.com/Srinivas26k/srvdb/blob/master/AGENTS.md) and
the workflow in
[CONTRIBUTING.md](https://github.com/Srinivas26k/srvdb/blob/master/CONTRIBUTING.md).
Everyone follows the
[Code of Conduct](https://github.com/Srinivas26k/srvdb/blob/master/CODE_OF_CONDUCT.md).

Contributions are licensed under Apache-2.0, section 5. There is no
contributor licence agreement in the repository today.

Report security issues privately, as described in
[SECURITY.md](https://github.com/Srinivas26k/srvdb/blob/master/SECURITY.md).
