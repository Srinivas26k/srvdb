"""Render VecCrash outcome counts from results.json.

    python benchmarks/veccrash/render.py

Writes veccrash.png (1600x900), veccrash-square.png (1200x1200) and
veccrash.pdf (vector) next to this file. Each library is one stacked bar
of outcome counts; SrvDB bars use the accent colour, others grey. Shade
encodes the outcome; durability violations are hatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ACCENT = "#5B6CF0"
GREY = "#8A8F98"
OUTCOMES = [
    ("OK", "OK: all acked data intact", 1.0, ""),
    ("EXTRA_PARTIAL", "Extra/partial unacked data (allowed)", 0.45, ""),
    ("LOST_ACKED", "Lost acknowledged data", 0.85, "///"),
    ("SILENT_CORRUPTION", "Silent corruption", 0.85, "xxx"),
    ("CORRUPT_OPEN", "Fails to reopen", 0.85, "..."),
]
LABELS = {
    "srvdb": "SrvDB (flat)",
    "srvdb-hnsw": "SrvDB (HNSW)",
    "chroma": "Chroma",
    "lancedb": "LanceDB",
    "faiss": "FAISS *",
    "faiss-atomic": "FAISS + atomic rename",
    "hnswlib": "hnswlib *",
    "usearch": "usearch *",
    "sqlite-vec": "sqlite-vec (WAL)",
}


def shade(colour: str, alpha: float) -> tuple:
    r, g, b = to_rgb(colour)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def draw(res: dict, size: tuple[float, float], dpi: int, path: Path) -> None:
    libs = list(res["libraries"])
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    fig.patch.set_facecolor("white")
    for row, lib in enumerate(libs):
        data = res["libraries"][lib]
        base = ACCENT if lib.startswith("srvdb") else GREY
        left = 0.0
        for key, _, alpha, hatch in OUTCOMES:
            n = data["counts"][key]
            if n:
                ax.barh(
                    row,
                    n,
                    left=left,
                    color=shade(base, alpha),
                    hatch=hatch,
                    edgecolor="#2B2B2B" if hatch else "white",
                    linewidth=0.6,
                )
            left += n
        v = data["violations"]
        ax.text(
            left * 1.01,
            row,
            f"{v}/{data['trials']} violations",
            va="center",
            fontsize=10,
            color="#B3261E" if v else "#333333",
        )
    ax.set_yticks(range(len(libs)), [LABELS.get(lib, lib) for lib in libs], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Trials (SIGKILL at a seeded random time, then reopen)", fontsize=11)
    ax.set_xlim(0, max(d["trials"] for d in res["libraries"].values()) * 1.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    env = res["environment"]
    ax.set_title(
        "VecCrash: outcome of process kills during writes and saves",
        fontsize=14,
        loc="left",
        pad=28,
    )
    ax.text(
        0,
        1.02,
        f"dim {res['config']['dim']}, batch {res['config']['batch']}, "
        f"commit {env['commit'][:10]}, {env['os']}.  * no documented crash-safety "
        "guarantee; typical application save pattern",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    handles = [
        Patch(
            facecolor=shade(GREY, a),
            hatch=h,
            edgecolor="#2B2B2B" if h else "white",
            label=lab,
        )
        for _, lab, a, h in OUTCOMES
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.09),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    draw(res, (16, 9), 100, HERE / "veccrash.png")
    draw(res, (12, 12), 100, HERE / "veccrash-square.png")
    draw(res, (10, 5.6), 100, HERE / "veccrash.pdf")


if __name__ == "__main__":
    main()
