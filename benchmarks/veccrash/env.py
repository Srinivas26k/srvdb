"""Environment capture and Wilson score intervals for VecCrash."""

from __future__ import annotations

import math
import os
import platform
import subprocess
from importlib import metadata

PACKAGES = {
    "srvdb": "srvdb",
    "chroma": "chromadb",
    "lancedb": "lancedb",
    "faiss": "faiss-cpu",
    "hnswlib": "hnswlib",
    "usearch": "usearch",
    "sqlite-vec": "sqlite-vec",
}


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson score 95% interval for k successes in n trials."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 5), round(min(1.0, centre + half), 5)]


def _licence(dist: str) -> str:
    md = metadata.metadata(dist)
    lic = md.get("License-Expression") or md.get("License") or ""
    if not lic or len(lic) > 60:
        classifiers = [
            c.split("::")[-1].strip()
            for c in md.get_all("Classifier") or []
            if c.startswith("License ::")
        ]
        lic = ", ".join(classifiers) or lic[:60]
    return lic


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f"unavailable ({type(e).__name__})"


def _cpu() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass  # non-Linux: fall back to platform.processor()
    return platform.processor()


def _fs(path: str) -> str:
    return _cmd(["stat", "-f", "-c", "%T", path])


def environment(workdir: str = "/tmp") -> dict:
    import sqlite3

    import numpy

    libs = {}
    for name, dist in PACKAGES.items():
        try:
            libs[name] = {
                "package": dist,
                "version": metadata.version(dist),
                "licence": _licence(dist),
            }
        except metadata.PackageNotFoundError:
            libs[name] = {"package": dist, "version": None, "licence": None}
    libs["sqlite-vec"]["sqlite_version"] = sqlite3.sqlite_version
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {
        "commit": _cmd(["git", "rev-parse", "HEAD"]),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": _cpu(),
        "cpus": os.cpu_count(),
        "ram_gib": round(mem, 1),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "workdir_filesystem": _fs(workdir),
        "libraries": libs,
    }
