"""Edge-deployment measurements for embedded vector stores.

    python benchmarks/edge/measure.py --venvs /data/build/veccrash/edge-venvs

Per library: wheel sizes from the PyPI JSON API (exact version installed
in a fresh venv), installed site-packages size and transitive package
count, peak RSS of an import-only process and of a create + add 1,000 x
128 process, and the same workload run with no network (``unshare -rn``,
a fresh unprivileged network namespace with only a down loopback) under
``strace`` to count outbound connection attempts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "veccrash"))
from env import environment  # noqa: E402
from static import STATIC  # noqa: E402
from workloads import IMPORTS, WORKLOADS  # noqa: E402

PACKAGES = {
    "srvdb": "srvdb",
    "chroma": "chromadb",
    "lancedb": "lancedb",
    "faiss": "faiss-cpu",
    "hnswlib": "hnswlib",
    "usearch": "usearch",
    "sqlite-vec": "sqlite-vec",
}
PLATFORMS = {
    "linux_x86_64": re.compile(r"manylinux.*x86_64"),
    "linux_aarch64": re.compile(r"manylinux.*aarch64"),
    "macos_arm64": re.compile(r"macosx.*(arm64|universal2)"),
    "win_amd64": re.compile(r"win_amd64"),
}
PY = "cp312"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def wheel_sizes(dist: str, version: str) -> dict:
    url = f"https://pypi.org/pypi/{dist}/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as r:
        files = json.load(r)["urls"]
    out = {}
    for plat, pat in PLATFORMS.items():
        cands = [
            f
            for f in files
            if f["filename"].endswith(".whl")
            and pat.search(f["filename"])
            and (
                f"-{PY}-" in f["filename"]
                or "-abi3-" in f["filename"]
                or "-py3-none-" in f["filename"]
            )
        ]
        if not cands:
            anyw = [f for f in files if f["filename"].endswith("-none-any.whl")]
            cands = anyw
        best = min(cands, key=lambda f: f["size"]) if cands else None
        out[plat] = (
            {"filename": best["filename"], "mib": round(best["size"] / 2**20, 2)}
            if best
            else None
        )
    sdists = [f for f in files if f["packagetype"] == "sdist"]
    out["sdist_mib"] = round(sdists[0]["size"] / 2**20, 2) if sdists else None
    out["prebuilt_linux_aarch64"] = out["linux_aarch64"] is not None
    return out


def dir_mib(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return round(total / 2**20, 1)


def make_venv(base: Path, name: str, dist: str) -> tuple[Path, dict]:
    venv = base / name
    shutil.rmtree(venv, ignore_errors=True)
    run(["uv", "venv", "-q", "-p", "3.12", "--seed", str(venv)], check=True)
    py = venv / "bin" / "python"
    before = len(
        json.loads(run([str(py), "-m", "pip", "list", "--format=json"]).stdout)
    )
    site = Path(
        run(
            [str(py), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"]
        ).stdout.strip()
    )
    base_mib = dir_mib(site)
    run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", dist], check=True)
    pkgs = json.loads(run([str(py), "-m", "pip", "list", "--format=json"]).stdout)
    version = next(p["version"] for p in pkgs if p["name"].lower() == dist.lower())
    info = {
        "version": version,
        "site_packages_mib": round(dir_mib(site) - base_mib, 1),
        "transitive_packages": len(pkgs) - before - 1,  # minus the library itself
        "packages": sorted(f"{p['name']}=={p['version']}" for p in pkgs),
    }
    # The workloads generate data with numpy; install it after measuring.
    run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", "numpy"], check=True)
    return py, info


def peak_rss(py: Path, code: str, cwd: Path, offline: bool) -> dict:
    """Run ``code`` in a fresh process; report peak RSS and connect attempts."""
    shutil.rmtree(cwd, ignore_errors=True)
    cwd.mkdir(parents=True)
    trace = cwd.parent / (cwd.name + ".strace")
    cmd = [str(py), "-c", code]
    if offline:
        cmd = [
            "unshare",
            "-rn",
            "strace",
            "-f",
            "-qq",
            "-e",
            "trace=connect",
            "-o",
            str(trace),
            *cmd,
        ]
    env = {
        k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")
    }
    r = run(cmd, cwd=cwd, env=env, timeout=600)
    res: dict = {"ok": r.returncode == 0}
    m = re.search(r"PEAK_RSS_KIB=(\d+)", r.stdout)
    if m:
        res["peak_rss_mib"] = round(int(m.group(1)) / 1024, 1)
    if r.returncode != 0:
        res["error"] = (
            r.stderr.strip().splitlines()[-1][:300] if r.stderr.strip() else "no stderr"
        )
    if offline and trace.exists():
        lines = trace.read_text().splitlines()
        net = [ln for ln in lines if "AF_INET" in ln]
        res["inet_connect_attempts"] = len(net)
        res["inet_connect_targets"] = sorted(
            {
                m.group(0)
                for ln in net
                for m in [re.search(r'inet_addr\("[^"]+"\)|inet_pton\([^)]*\)', ln)]
                if m
            }
        )[:10]
        trace.unlink()
    shutil.rmtree(cwd, ignore_errors=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venvs", default="/data/build/veccrash/edge-venvs")
    ap.add_argument("--libs", default=",".join(PACKAGES))
    ap.add_argument("--keep-venvs", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()
    base = Path(a.venvs)
    base.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    results = json.loads(out.read_text()) if out.exists() else {"libraries": {}}
    results["command"] = "python " + " ".join(sys.argv)
    env = environment(str(base))
    env.pop("libraries", None)  # versions are per fresh venv below
    results["environment"] = env
    results["offline_method"] = (
        "unshare -rn (new unprivileged user + network namespace, no interfaces up); "
        "strace -f -e trace=connect counts AF_INET/AF_INET6 connect() attempts"
    )
    for lib in a.libs.split(","):
        dist = PACKAGES[lib]
        py, info = make_venv(base, lib, dist)
        info["wheels"] = wheel_sizes(dist, info["version"])
        work = base / f"{lib}-work"
        info["import_only"] = peak_rss(py, IMPORTS[lib], work, offline=False)
        info["create_add_1000"] = peak_rss(py, WORKLOADS[lib], work, offline=False)
        info["offline_create_add_1000"] = peak_rss(
            py, WORKLOADS[lib], work, offline=True
        )
        info["documented"] = STATIC[lib]
        results["libraries"][lib] = info
        print(
            lib,
            json.dumps({k: v for k, v in info.items() if k != "packages"}),
            flush=True,
        )
        out.write_text(json.dumps(results, indent=2) + "\n")
        if not a.keep_venvs:
            shutil.rmtree(base / lib)


if __name__ == "__main__":
    main()
