"""Child processes for run.py: one per dataset preparation, one per mode.

    python measure.py prepare <dataset>          cache arrays as .npy
    python measure.py mode <dataset> <mode-key>  build, measure, print JSON

Each mode runs in its own process so peak RSS is per mode and memory is
freed between runs. Nothing here measures time-dependent speed.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import DATA, prepare  # noqa: E402

K = 10
BATCH = 10_000
# key -> (index name, options, rerank override applied after build)
MODES = {
    "flat": ("flat", {}, None),
    "sq8": ("sq8", {}, None),
    "pq": ("pq", {}, None),
    "rabitq4": ("rabitq", {"bits": 4}, None),
    "rabitq2": ("rabitq", {"bits": 2}, None),
    "rabitq1": ("rabitq", {"bits": 1}, None),
    "rabitq4-norerank": ("rabitq", {"bits": 4}, 0),
    "rabitq2-norerank": ("rabitq", {"bits": 2}, 0),
    "rabitq1-norerank": ("rabitq", {"bits": 1}, 0),
}
# On-disk files by role. Everything else (lock, crc, settings) is "other".
ROLE = {
    "vectors.bin": "full_vectors",
    "sq8_vectors.bin": "codes",
    "quantized_vectors.bin": "codes",
    "rabitq_codes.bin": "codes",
    "scalar_quantizer.bin": "quantizer",
    "codebooks.bin": "quantizer",
    "rabitq_params.bin": "quantizer",
    "metadata.db": "ids_metadata",
}


def cache_dir(name: str) -> Path:
    return DATA / "cache" / name


def do_prepare(name: str) -> None:
    out = cache_dir(name)
    if not (out / "prov.json").exists():
        base, q, gt, prov = prepare(name, K)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "base.npy", base)
        np.save(out / "queries.npy", q)
        np.save(out / "gt.npy", gt)
        (out / "prov.json").write_text(json.dumps(prov, indent=1))
    print((out / "prov.json").read_text())


def disk_bytes(path: Path) -> dict:
    roles: dict = {}
    for f in path.rglob("*"):
        if f.is_file():
            role = ROLE.get(f.name, "other")
            roles[role] = roles.get(role, 0) + f.stat().st_size
    return roles


def do_mode(name: str, key: str) -> None:
    import srvdb

    src = cache_dir(name)
    base = np.load(src / "base.npy", mmap_mode="r")
    q = np.load(src / "queries.npy")
    gt = np.load(src / "gt.npy")
    n, dim = base.shape
    index, opts, rerank = MODES[key]
    path = DATA / "work" / f"{name}-{key}"
    if path.exists():
        srvdb.delete(str(path))
    db = srvdb.create(str(path), dim, index=index, **opts)
    for i in range(0, n, BATCH):
        chunk = np.ascontiguousarray(base[i : i + BATCH])
        db.add([str(j) for j in range(i, i + len(chunk))], chunk, ["{}"] * len(chunk))
    db.save()
    if rerank is not None:
        db.set_rabitq_rerank(rerank)
    hits = db.search_batch(q, K)
    found = [{int(i) for i, _ in row} for row in hits]
    recall = float(np.mean([len(f & set(g.tolist())) / K for f, g in zip(found, gt)]))
    report = db.memory_report()
    db.close()
    roles = disk_bytes(path)
    srvdb.delete(str(path))
    if path.exists():
        shutil.rmtree(path)
    total = sum(roles.values())
    print(
        json.dumps(
            {
                "mode": key,
                "index": index,
                "options": opts,
                "rerank_override": rerank,
                "n": int(n),
                "dim": int(dim),
                "queries": len(q),
                "k": K,
                "recall_at_10": round(recall, 4),
                "disk_bytes_by_role": roles,
                "disk_bytes_total": total,
                "disk_bytes_per_vector_total": round(total / n, 2),
                # The bytes the scan reads per vector: codes, or float32 for flat.
                "disk_bytes_per_vector_search_codes": round(
                    roles.get("codes", roles.get("full_vectors", 0)) / n, 2
                ),
                "memory_report": report,
                "memory_report_bytes_per_vector": round(
                    (report["vector_bytes"] + report["index_bytes"]) / n, 2
                ),
                "peak_rss_mib": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
                ),
                "input_arrays_mib": round((base.nbytes + q.nbytes) / 2**20, 1),
            }
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 1))
    if sys.argv[1] == "prepare":
        do_prepare(sys.argv[2])
    else:
        do_mode(sys.argv[2], sys.argv[3])
