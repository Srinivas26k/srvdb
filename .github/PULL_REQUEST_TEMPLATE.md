## What and why

<!-- What does this change, and why? Link the issue: "Fixes #123". -->

## How it was tested

<!-- Bug fixes: the test that failed before the fix. Storage changes: a reopen test. -->

## Checklist

- [ ] `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` pass
- [ ] `pytest tests/python` passes (if Python or bindings changed)
- [ ] Docs updated: website page, `_low_level.pyi`, `CHANGELOG.md` (if user-visible)
- [ ] Follows [AGENTS.md](../AGENTS.md) (300-line limit, typed errors, no unwrap on user input)
