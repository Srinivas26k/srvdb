"""Environment block: hardware, OS, SrvDB commit and library versions."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata

LIBRARIES = ["srvdb", "faiss-cpu", "hnswlib", "usearch", "turbovec", "numpy"]


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def _physical_cores() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


def _os() -> str:
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    name = line.split("=", 1)[1].strip().strip('"')
                    return f"{name} (kernel {platform.release()})"
    except OSError:
        pass
    return f"{platform.system()} {platform.release()}"


def _commit(repo: str) -> str:
    def git(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", repo, *a], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short=12", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment(repo: str, command: str) -> dict:
    return {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "ram_gib": _ram_gib(),
        "os": _os(),
        "python": platform.python_version(),
        "commit": _commit(repo),
        "versions": versions(),
        "command": command,
    }
