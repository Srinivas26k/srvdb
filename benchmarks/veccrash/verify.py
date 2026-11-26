"""VecCrash verifier: reopen a killed store in a fresh process and classify.

Prints one JSON object to stdout. Outcomes, in priority order:

- CORRUPT_OPEN: reopening or reading raised an error (the parent also
  uses this when the verifier itself crashes or hangs);
- SILENT_CORRUPTION: the store opened but a visible id is unknown, or a
  visible vector differs from what was written (acknowledged or not);
- LOST_ACKED: an acknowledged id is missing;
- EXTRA_PARTIAL: unacknowledged data is visible (whole or partial
  batches). Allowed; not a durability violation;
- OK: exactly the acknowledged batches are visible and intact.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from adapters import ADAPTERS
from writer import batch_ids, batch_vectors

ATOL = 1e-6  # float tolerance; every library here stores float32 exactly


def read_acks(path: str) -> set[int]:
    with open(path, "rb") as f:
        lines = f.read().decode(errors="replace").split("\n")
    # A torn last line cannot occur for a completed write, but be strict.
    return {int(x) for x in lines if x.strip().isdigit()}


def expected(seed: int, batches: int, batch: int, dim: int) -> dict[int, np.ndarray]:
    exp: dict[int, np.ndarray] = {}
    for b in range(batches):
        exp.update(zip(batch_ids(b, batch), batch_vectors(seed, b, batch, dim)))
    return exp


def classify(store, acked, exp, batch, values_ok) -> dict:
    acked_ids = {i for b in acked for i in batch_ids(b, batch)}
    unknown = [i for i in store if i not in exp]
    wrong = [i for i in store if i in exp and not values_ok(i)]
    missing = sorted(acked_ids - set(store))
    extra = sorted(set(store) - acked_ids - set(unknown))
    extra_batches = {(i - 1) // batch for i in extra}
    partial = [b for b in extra_batches if not set(batch_ids(b, batch)) <= set(store)]
    if unknown or wrong:
        outcome = "SILENT_CORRUPTION"
    elif missing:
        outcome = "LOST_ACKED"
    elif extra:
        outcome = "EXTRA_PARTIAL"
    else:
        outcome = "OK"
    return {
        "outcome": outcome,
        "acked_batches": len(acked),
        "visible": len(store),
        "missing_acked": len(missing),
        "missing_acked_batches": sorted({(i - 1) // batch for i in missing}),
        "wrong_values": len(wrong),
        "unknown_ids": len(unknown),
        "extra_ids": len(extra),
        "partial_batches": len(partial),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ack", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--batches", type=int, default=40)
    a = ap.parse_args()

    acked = read_acks(a.ack)
    exp = expected(a.seed, a.batches, a.batch, a.dim)
    lib = ADAPTERS[a.lib]()
    try:
        store = lib.read_all(a.dir, a.dim)
        if hasattr(lib, "self_search"):
            present = {i: exp[i] for i, v in store.items() if i in exp and v.size == 0}
            hits = lib.self_search(a.dir, present)
            values_ok = lambda i: hits.get(i, False)  # noqa: E731
        else:
            values_ok = lambda i: bool(  # noqa: E731
                store[i].shape == exp[i].shape
                and np.allclose(store[i], exp[i], rtol=0, atol=ATOL)
            )
        res = classify(store, acked, exp, a.batch, values_ok)
    except Exception as e:  # any failure to open or read is the finding
        res = {
            "outcome": "CORRUPT_OPEN",
            "acked_batches": len(acked),
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        }
    sys.stdout.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
