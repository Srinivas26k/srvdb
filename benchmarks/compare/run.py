"""Compare SrvDB with FAISS, hnswlib, usearch and turbovec on the same data.

Run from the repository root inside the benchmark venv (see README.md):

    python benchmarks/compare/run.py --dims 384,768 --ns 5000,50000

Each (dataset, library/index, thread count) runs in its own child process so
peak RSS is per index and thread limits apply from process start.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from child import REGISTRY  # noqa: E402
from data import prepare  # noqa: E402
from envinfo import environment  # noqa: E402

THREAD_ENV = ["OMP_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dims", type=ints, default=[384, 768])
    p.add_argument("--ns", type=ints, default=[5000, 50000])
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", default="1,all", help="comma list; 'all' = os.cpu_count()")
    p.add_argument("--libs", default=",".join(REGISTRY), help="comma list of " + ",".join(REGISTRY))
    p.add_argument("--work-dir", default=None, help="datasets and temporary indexes")
    p.add_argument("--preliminary", action="store_true", help="mark results as not fit to publish")
    p.add_argument("--out", default=os.path.join(REPO, "website/src/data/benchmarks/comparison.json"))
    return p.parse_args()


def loadavg() -> list[float]:
    try:
        with open("/proc/loadavg", encoding="utf-8") as f:
            return [float(x) for x in f.read().split()[:3]]
    except OSError:
        return list(os.getloadavg())


def run_child(name: str, data_dir: str, threads: int, k: int, work: str) -> dict:
    env = dict(os.environ, **{v: str(threads) for v in THREAD_ENV})
    cmd = [sys.executable, os.path.join(HERE, "child.py"), name, data_dir, str(threads), str(k), work]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=HERE)
    if out.returncode != 0:
        tail = out.stderr.strip().splitlines()[-1:] or ["no output"]
        return {"name": name, "threads": threads, "error": tail[0]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    a = parse()
    cores = os.cpu_count() or 1
    threads = sorted({cores if t == "all" else int(t) for t in a.threads.split(",")})
    libs = [x for x in a.libs.split(",") if x]
    unknown = [x for x in libs if x not in REGISTRY]
    if unknown:
        sys.exit(f"unknown library: {unknown}; choose from {list(REGISTRY)}")
    work = a.work_dir or tempfile.mkdtemp(prefix="srvdb-compare-")
    os.makedirs(work, exist_ok=True)

    command = "python benchmarks/compare/run.py " + shlex.join(sys.argv[1:])
    doc = {
        "preliminary": a.preliminary,
        "environment": environment(REPO, command.strip()),
        "parameters": {
            "k": a.k, "threads": threads, "hnsw_m": 16, "hnsw_ef_construction": 200,
            "ef_search_sweep": __import__("adapters").EF_SWEEP,
            "nprobe_sweep": __import__("adapters").NPROBE_SWEEP,
            "ivf_nlist": "sqrt(n)", "srvdb_ivf_nprobe": 8, "warmup_queries": 10,
            "qps": "one query per call on one caller thread, after warm-up",
            "batch_qps": "all queries in one batch call",
        },
        "runs": [],
    }
    for dim in a.dims:
        for n in a.ns:
            data_dir, ds = prepare(REPO, work, dim, n, a.queries, a.seed, a.k)
            results = []
            for name in libs:
                for t in threads:
                    print(f"dim={dim} n={n} {name} threads={t} ...", file=sys.stderr, flush=True)
                    before = loadavg()
                    if before[0] > 1.0:
                        print(f"WARNING: 1-minute load {before[0]:.2f} > 1; QPS and latency are not publishable",
                              file=sys.stderr, flush=True)
                    r = run_child(name, data_dir, t, a.k, work)
                    r["loadavg_before"], r["loadavg_after"] = before, loadavg()
                    results.append(r)
            doc["runs"].append({"dataset": ds, "results": results})
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
                f.write("\n")
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
