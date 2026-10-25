---
title: Benchmarks
description: Reproducible performance numbers (coming soon).
---

:::note[Coming soon]
SrvDB publishes a performance number only when it comes from a committed,
reproducible benchmark that lists the hardware, OS, dataset, commit and
command. That suite is being built for 0.4.
:::

Numbers in earlier READMEs measured versions where some modes did not work as
labelled, so they were withdrawn.

Meanwhile, measure on your own hardware:

```bash
pip install "srvdb[bench]"
python benchmarks/universal_benchmark.py
```

Share results (with hardware, OS, version and command) in
[GitHub Discussions](https://github.com/Srinivas26k/srvdb/discussions).
