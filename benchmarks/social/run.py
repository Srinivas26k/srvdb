"""Measure SrvDB bytes per vector and recall@10 per mode, then render charts.

    python benchmarks/social/run.py [--datasets a,b] [--crash] [--no-render]

Datasets run one at a time; every mode runs in its own subprocess, and
results.json is rewritten after each mode so partial results survive.
Speed (QPS, latency, build time) is deliberately not measured here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from datasets import DATASETS  # noqa: E402
from measure import MODES  # noqa: E402

OUT = HERE / "results.json"
ORDER = [
    "nytimes-256-angular",
    "glove-100-angular",
    "dbpedia-openai-100k",
    "synthetic-768",
]


def _cmd(*args: str) -> str:
    return subprocess.run(
        args, capture_output=True, text=True, check=True, cwd=ROOT
    ).stdout.strip()


def environment(argv: list[str]) -> dict:
    import srvdb

    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    mem = next(
        line.split()[1]
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal")
    )
    dirty = _cmd("git", "status", "--porcelain", "--untracked-files=no")
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_gib": round(int(mem) / 2**20, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "srvdb_version": getattr(srvdb, "__version__", None) or _version(),
        "git_commit": _cmd("git", "rev-parse", "HEAD"),
        "git_tree_dirty": bool(dirty),
        "command": "python " + " ".join(["benchmarks/social/run.py", *argv]),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def child(*args: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "measure.py"), *args],
        stdout=subprocess.PIPE,  # stderr passes through
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def crash_test() -> dict:
    """Reproduce the CI kill -9 test locally (ADR 0003) and parse its counts."""
    iterations = "250"
    env = os.environ | {"SRVDB_CRASH_ITERATIONS": iterations}
    cmd = [
        "cargo",
        "test",
        "--release",
        "--locked",
        "-p",
        "srvdb",
        "--test",
        "test_crash",
        "--",
        "--nocapture",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    found = re.findall(
        r"^(\w+): (\d+) kills, (\d+) committed vectors verified$",
        run.stderr + run.stdout,
        re.M,
    )
    return {
        "command": f"SRVDB_CRASH_ITERATIONS={iterations} " + " ".join(cmd),
        "passed": run.returncode == 0,
        "per_mode": {
            m: {"kills": int(k), "committed_verified": int(v)} for m, k, v in found
        },
        "kills_total": sum(int(k) for _, k, _ in found),
        "committed_lost": 0 if run.returncode == 0 else None,
    }


def save(results: dict) -> None:
    OUT.write_text(json.dumps(results, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(ORDER))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--crash", action="store_true", help="also run the kill -9 test")
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args()
    results = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    results["environment"] = environment(sys.argv[1:])
    for name in a.datasets.split(","):
        if name not in DATASETS:
            sys.exit(f"unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
        start = time.time()
        prov = child("prepare", name)
        entry = {
            "provenance": prov,
            "prepare_seconds": round(time.time() - start, 1),
            "modes": {},
        }
        results["datasets"][name] = entry
        save(results)
        for key in a.modes.split(","):
            r = child("mode", name, key)
            entry["modes"][key] = r
            save(results)
            print(
                f"{name} {key}: recall@10 {r['recall_at_10']}, "
                f"{r['disk_bytes_per_vector_search_codes']} B/vec codes, "
                f"peak RSS {r['peak_rss_mib']} MiB",
                flush=True,
            )
    if a.crash:
        results["crash_test"] = crash_test()
        save(results)
    if not a.no_render:
        subprocess.run([sys.executable, str(HERE / "render.py")], check=True)


if __name__ == "__main__":
    main()
