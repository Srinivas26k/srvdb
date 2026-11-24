"""Edge and small-memory devices: footprint, recall and deadlines.

Compares the encodings that fit a small device (flat float32, SQ8 and
RaBitQ at 4, 2 and 1 bits) on the same dataset: estimated resident
memory from ``db.memory_report()``, bytes on disk, recall@10 and latency.
It then shows two RaBitQ features for devices with tight time limits:
``certify=True`` (rescore only candidates the error bounds cannot settle)
and ``deadline_ms=`` (stop the scan in time and report completeness).
Finally it copies a database the way a deployment script might, without
its ``.crc`` checksum files, and repairs it with ``srvdb.rebuild_checksums``.
The chart plots memory against recall, one point per encoding.

Run::

    python examples/gallery/12_edge/edge.py [--fast]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402

import _common  # noqa: E402
import srvdb  # noqa: E402

DIM = 128
K = 10
ENCODINGS = {
    "flat": ("flat", {}),
    "sq8": ("sq8", {}),
    "rabitq 4-bit": ("rabitq", {"bits": 4}),
    "rabitq 2-bit": ("rabitq", {"bits": 2}),
    "rabitq 1-bit": ("rabitq", {"bits": 1}),
}


def disk_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def evaluate(db, queries, expected, **kw):
    found, scanned = [], []
    with _common.Timer() as t:
        for q in queries.tolist():
            out = db.search(q, k=K, **kw)
            found.append([hit[0] for hit in out])
            if kw:
                scanned.append(out.completeness)
    ms = 1000 * t.seconds / len(queries)
    return _common.recall_at_k(found, expected), ms, scanned


def main() -> None:
    args = _common.parse_args(__doc__.splitlines()[0])
    _common.print_environment("12 edge and small memory")
    n, nq = (4000, 40) if args.fast else (50000, 200)
    data, _ = make_blobs(n + nq, n_features=DIM, centers=80, random_state=12)
    data = data.astype(np.float32)
    data, queries = data[:n], data[n:]
    expected = [[str(j) for j in r] for r in _common.exact_top_k(data, queries, K)]
    ids = [str(i) for i in range(n)]
    print(f"dataset: make_blobs, {n} vectors, {nq} queries, dim {DIM}\n")

    rows = []
    with _common.scratch_dir() as tmp:
        for name, (index, opts) in ENCODINGS.items():
            path = tmp / name.replace(" ", "-")
            with srvdb.create(str(path), dimension=DIM, index=index, **opts) as db:
                db.add(ids, data.tolist(), ["{}"] * n)
                db.save()
                mem = db.memory_report()["total_bytes"]
                recall, ms, _ = evaluate(db, queries, expected)
                if index == "rabitq" and opts["bits"] == 1:
                    for kw in ({"certify": True}, {"deadline_ms": 0.05}):
                        r, m, comp = evaluate(db, queries, expected, **kw)
                        print(
                            f"  1-bit with {kw}: recall@{K}={r:.3f}  "
                            f"{m:.3f} ms/query  mean completeness={np.mean(comp):.2f}"
                        )
            rows.append((name, mem, disk_bytes(path), recall, ms))
            print(
                f"{name:<13} memory={mem / 2**20:6.2f} MiB  "
                f"disk={disk_bytes(path) / 2**20:6.2f} MiB  "
                f"recall@{K}={recall:.3f}  {ms:.3f} ms/query"
            )

        # Deployment copy that forgot the checksum files.
        src, dst = tmp / "rabitq-4-bit", tmp / "device-copy"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.crc"))
        try:
            srvdb.open(str(dst)).close()
            print("\ncopy opened without checksums")
        except (OSError, RuntimeError) as err:
            print(f"\nopen without .crc files failed: {type(err).__name__}")
        print(f"rebuild_checksums rewrote: {srvdb.rebuild_checksums(str(dst))}")
        with srvdb.open(str(dst)) as db:
            print(f"reopened copy holds {db.count()} records")

    plt = _common.pyplot()
    fig, ax = plt.subplots()
    ax.scatter([r[1] / 2**20 for r in rows], [r[3] for r in rows], color="#111111")
    for name, mem, _, recall, _ in rows:
        ax.annotate(
            name,
            (mem / 2**20, recall),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("estimated resident memory (MiB, log scale)")
    ax.set_ylabel(f"recall@{K}")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Memory vs recall, {n} vectors x {DIM} dims (this run)")
    _common.save_chart(fig, args.out, "12_edge.png")


if __name__ == "__main__":
    main()
