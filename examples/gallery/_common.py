"""Shared helpers for the SrvDB examples gallery.

Every gallery script uses these to print the environment (hardware, OS,
library versions, commit), to parse ``--fast`` and ``--out``, to make a
scratch directory for databases and to save monochrome charts. Nothing
here makes network calls.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

GALLERY = Path(__file__).resolve().parent
DEFAULT_OUT = GALLERY / "output"


def parse_args(description: str) -> argparse.Namespace:
    """``--fast`` shrinks the run for CI; ``--out`` sets the chart directory."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--fast", action="store_true", help="small sizes, for CI and smoke tests"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="directory for charts"
    )
    return parser.parse_args()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=GALLERY,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass  # not Linux: fall back to platform.processor()
    return platform.processor() or platform.machine()


def version(module: str) -> str:
    try:
        return str(importlib.import_module(module).__version__)
    except ImportError:
        return "not installed"


def print_environment(title: str, extra: tuple[str, ...] = ()) -> None:
    """Print what produced the numbers below (AGENTS.md section 11)."""
    print(f"== {title} ==")
    print(f"hardware: {cpu_model()}, {os.cpu_count()} logical CPUs")
    print(f"os:       {platform.system()} {platform.release()}")
    print(f"python:   {platform.python_version()}")
    libs = ["srvdb", "numpy", "sklearn", "matplotlib", *extra]
    print("versions: " + ", ".join(f"{m} {version(m)}" for m in libs))
    print(f"commit:   {git_commit()}")
    print()


@contextmanager
def scratch_dir() -> Iterator[Path]:
    """A temporary parent directory; put databases in children of it."""
    with tempfile.TemporaryDirectory(prefix="srvdb-gallery-") as tmp:
        yield Path(tmp)


def normalize(x: np.ndarray) -> np.ndarray:
    """Unit-length rows (SrvDB scores are cosine similarities)."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def exact_top_k(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground truth: indices of the k most cosine-similar rows per query."""
    sims = normalize(queries) @ normalize(data).T
    return np.argsort(-sims, axis=1)[:, :k]


def recall_at_k(found: list[list[str]], truth: list[list[str]]) -> float:
    """Fraction of true neighbours that were returned."""
    hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth))
    return hits / max(1, sum(len(t) for t in truth))


class Timer:
    """``with Timer() as t: ...`` then read ``t.seconds``."""

    seconds = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start


def meta(**fields: object) -> str:
    """Metadata JSON for ``db.add()``."""
    return json.dumps(fields)


def pyplot():
    """matplotlib.pyplot with a headless backend and the docs' monochrome look."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.2),
            "figure.dpi": 110,
            "font.size": 10,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.prop_cycle": matplotlib.cycler(
                color=["#111111", "#666666", "#999999", "#bbbbbb", "#333333"]
            ),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.8,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "image.cmap": "Greys",
        }
    )
    return plt


def save_chart(fig, out: Path, name: str) -> Path:
    """Save ``fig`` as ``out/name``, close it and print the path."""
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    print(f"chart: {path}")
    return path
