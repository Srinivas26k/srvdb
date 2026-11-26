"""Render the edge-footprint comparison from results.json.

    python benchmarks/edge/render.py

Writes edge.png (1600x900), edge-square.png (1200x1200) and edge.pdf next
to this file: three panels (wheel download size for linux x86_64,
installed site-packages size, peak RSS to create a store and add 1,000
vectors), log scale, SrvDB in the accent colour and others grey.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ACCENT = "#5B6CF0"
GREY = "#8A8F98"
NAMES = {
    "srvdb": "SrvDB",
    "chroma": "Chroma",
    "lancedb": "LanceDB",
    "faiss": "FAISS",
    "hnswlib": "hnswlib",
    "usearch": "usearch",
    "sqlite-vec": "sqlite-vec",
}


def metrics(info: dict) -> list[float | None]:
    wheel = (info["wheels"].get("linux_x86_64") or {}).get("mib")
    rss = info["create_add_1000"].get("peak_rss_mib")
    return [wheel, info["site_packages_mib"], rss]


PANELS = [
    "Wheel download, linux x86_64 (MiB)",
    "Installed size incl. dependencies (MiB)",
    "Peak RSS: create + add 1,000 x 128 (MiB, incl. numpy)",
]


def draw(res: dict, size: tuple[float, float], path: Path, cols: int) -> None:
    libs = list(res["libraries"])
    rows = (len(PANELS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=size, dpi=100, squeeze=False)
    fig.patch.set_facecolor("white")
    for p, title in enumerate(PANELS):
        ax = axes[p // cols][p % cols]
        vals = [metrics(res["libraries"][lib])[p] for lib in libs]
        colours = [ACCENT if lib == "srvdb" else GREY for lib in libs]
        ax.barh(range(len(libs)), [v or 0 for v in vals], color=colours)
        log = p < 2  # sizes span three orders of magnitude; RSS does not
        if log:
            ax.set_xscale("log")
        ax.set_yticks(
            range(len(libs)), [NAMES.get(lib, lib) for lib in libs], fontsize=11
        )
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, loc="left")
        top = max(v or 0 for v in vals)
        for i, v in enumerate(vals):
            x = (v or 0.1) * 1.08 if log else (v or 0) + top * 0.01
            ax.text(
                x,
                i,
                "no wheel (sdist only)" if v is None else f"{v:g}",
                va="center",
                fontsize=9.5,
            )
        ax.set_xlim(*((0.1, top * 4) if log else (0, top * 1.2)))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for q in range(len(PANELS), rows * cols):
        axes[q // cols][q % cols].axis("off")
    env = res["environment"]
    fig.suptitle(
        "Edge footprint of embedded vector stores (lower is better; size panels on log scale)",  # noqa: E501
        fontsize=14,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        f"Python {env['python']}, {env['os']}, commit {env['commit'][:10]}. "
        "Installed size includes every transitive dependency pulled in by pip.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    draw(res, (16, 9), HERE / "edge.png", cols=3)
    draw(res, (12, 12), HERE / "edge-square.png", cols=2)
    draw(res, (12, 5), HERE / "edge.pdf", cols=3)


if __name__ == "__main__":
    main()
