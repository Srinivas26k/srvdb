"""VecCrash writer child process.

Adds batches of seeded vectors with known ids to one library. After each
batch the library's durability point is called (save, commit or write);
only after it returns is the batch number appended to the ack log and the
ack log fsynced. The ack log is therefore the exact set of batches the
library promised were durable. The parent SIGKILLs this process at a
random time.

Batch 0 is written and acknowledged before the ready file is created, so
every trial starts with a non-empty promised state.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from adapters import ADAPTERS


def batch_ids(b: int, batch: int) -> list[int]:
    """Ids of batch ``b``: a contiguous block, starting at 1."""
    return list(range(1 + b * batch, 1 + (b + 1) * batch))


def batch_vectors(seed: int, b: int, batch: int, dim: int) -> np.ndarray:
    """Deterministic float32 vectors of batch ``b`` for the trial seed."""
    rng = np.random.default_rng([seed, b])
    return rng.standard_normal((batch, dim), dtype=np.float32)


def ack(fd: int, b: int) -> None:
    os.write(fd, f"{b}\n".encode())
    os.fsync(fd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ack", required=True)
    ap.add_argument("--ready", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--batches", type=int, default=40)
    a = ap.parse_args()

    lib = ADAPTERS[a.lib]()
    handle = lib.open_writer(a.dir, a.dim)
    fd = os.open(a.ack, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    for b in range(a.batches):
        lib.add(handle, batch_ids(b, a.batch), batch_vectors(a.seed, b, a.batch, a.dim))
        lib.durable(handle)
        ack(fd, b)
        if b == 0:
            with open(a.ready, "w") as f:
                f.write("ready\n")
    os.close(fd)
    # Exit without closing the store: a clean close could flush state the
    # library did not promise at the durability point.
    os._exit(0)


if __name__ == "__main__":
    main()
