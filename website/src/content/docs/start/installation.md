---
title: Installation
description: Install SrvDB from PyPI or build it from source.
---

## Python

```bash
pip install srvdb
```

Prebuilt wheels are published for **Python 3.9 to 3.14** on:

| OS | Architectures |
|----|---------------|
| Linux (manylinux) | x86_64, aarch64 |
| macOS | x86_64, arm64 (Apple Silicon) |
| Windows | x64 |

No Rust toolchain is needed for wheels. SrvDB has **no Python runtime
dependencies**.

To run the benchmark script you also need numpy, psutil and scikit-learn:

```bash
pip install "srvdb[bench]"
```

## From source

Needed on platforms without a wheel, or to develop SrvDB. Requires **Rust
1.99 or newer** and [maturin](https://www.maturin.rs/).

```bash
rustup update stable
git clone https://github.com/Srinivas26k/srvdb
cd srvdb
pip install maturin
maturin develop --release     # builds and installs into the active venv
```

:::caution
Always build with `--release` for real use. Debug builds are much slower.
:::

## Rust

SrvDB is not on crates.io yet. Use it as a git dependency:

```toml
[dependencies]
srvdb = { git = "https://github.com/Srinivas26k/srvdb" }
```

See [Using SrvDB from Rust](/srvdb/guides/rust/).

## Check the install

```python
import srvdb
print(srvdb.__version__)
```
