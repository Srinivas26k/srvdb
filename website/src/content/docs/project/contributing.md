---
title: Contributing
description: How to contribute to SrvDB.
---

Contributions of every size are welcome: bug reports, docs fixes, tests,
benchmarks on your hardware, and code.

## Set up

```bash
git clone https://github.com/Srinivas26k/srvdb
cd srvdb
cargo test --workspace
pip install maturin pytest
maturin develop
pytest tests/python
```

## Before opening a pull request

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
pytest tests/python
```

- One logical change per pull request, with a test. Bug fixes include a test
  that fails without the fix.
- Commit messages follow Conventional Commits: `fix(search): ...`.
- Storage changes need a reopen test; durability changes a crash test.

Read [CONTRIBUTING.md](https://github.com/Srinivas26k/srvdb/blob/master/CONTRIBUTING.md)
and the engineering rules in
[AGENTS.md](https://github.com/Srinivas26k/srvdb/blob/master/AGENTS.md).
Report security issues privately to srinivasvarma764@gmail.com.
