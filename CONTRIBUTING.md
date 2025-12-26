# Contributing to SrvDB

We welcome contributions to SrvDB. To maintain the stability and performance of the database, please adhere to the following guidelines.

## Development Environment
1.  **Rust**: Ensure you have the latest stable Rust toolchain (`rustup update`).
2.  **Python**: Python 3.10+ required for binding tests.
3.  **Dependencies**: Install `maturin` for building Python wheels (`pip install maturin`).

## Pull Request Process
1.  **Performance Regression**: Any PR affecting the search or storage modules must include benchmark results (`python bench_release.py`) proving no regression in latency (>15ms) or throughput (<500 vecs/sec).
2.  **Test Coverage**: Run the full test suite via `cargo test` before submitting.
3.  **Formatting**: Ensure code is formatted with `cargo fmt`.

## Reporting Issues
Please include your OS, CPU architecture (for SIMD debugging), and a minimal reproduction script when filing issues.