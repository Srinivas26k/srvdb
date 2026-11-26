"""Recall@10 vs code bytes per vector: SrvDB against competitor quantizers.

Called by render.py; reads only the results.json entries passed in.
SrvDB points are accent-coloured, competitors grey with distinct markers.
"""

from __future__ import annotations

import charts
from charts import ACCENT, GREY, INK

SRVDB = [
    ("sq8", "SQ8"),
    ("pq", "PQ"),
    ("rabitq4", "RaBitQ 4-bit"),
    ("rabitq2", "RaBitQ 2-bit"),
    ("rabitq1", "RaBitQ 1-bit"),
]
# Competitor family -> (grey shade, marker)
STYLE = {
    "IndexRaBitQ": ("#555B66", "s"),
    "IndexScalarQuantizer": ("#8A8F99", "D"),
    "IndexPQ": ("#A3A7B0", "v"),
    "OPQ": ("#A3A7B0", "^"),
    "IndexBinaryFlat": ("#70757F", "X"),
    "TurboQuantIndex": ("#3E434C", "P"),
}


def _style(index: str) -> tuple[str, str]:
    return next((v for k, v in STYLE.items() if index.startswith(k)), (GREY, "o"))


def draw(env, name, nice, ds, size, footer, out) -> None:
    modes, comp = (
        ds["modes"],
        [c for c in ds.get("competitors", []) if "skipped" not in c],
    )
    fig, band = charts.figure(
        size,
        f"Recall@10 vs bytes per vector: {nice}",
        "Same data, queries and ground truth. SrvDB in blue (default rerank), "
        "FAISS and turbovec in grey: filled = float32 rerank, hollow = none.",
        footer,
    )
    top, bottom = band
    sq = size == "square"
    if sq and comp:
        bottom += 0.24  # room for the key under the plot
    ax = fig.add_axes((0.08, bottom, 0.86 if sq else 0.62, top - bottom))
    ax.set_xscale("log", base=2)
    for key, label in SRVDB:
        if key not in modes:
            continue
        m = modes[key]
        x, y = m["memory_report"]["vector_bytes"] / m["n"], m["recall_at_10"]
        ax.scatter(x, y, s=160, color=ACCENT, zorder=4, edgecolor="white")
        ax.annotate(
            label,
            (x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=13,
            color=ACCENT,
            fontweight="bold",
        )
    for c in comp:
        shade, marker = _style(c["index"])
        filled = c.get("rerank_k_factor")
        ax.scatter(
            c["code_bytes_per_vector"],
            c["recall_at_10"],
            s=110,
            marker=marker,
            color=shade if filled else "white",
            edgecolor=shade,
            lw=1.8,
            zorder=3,
            label=f"{c['library']} {c['index']}",
        )
    ax.set_xlabel("code bytes per vector (log scale)", color=GREY, fontsize=13)
    ax.set_ylabel("recall@10", color=GREY, fontsize=13)
    ax.set_ylim(0, 1.04)
    ax.grid(color="#ECEDF1", lw=1, zorder=0)
    ax.tick_params(colors=GREY, labelsize=12, length=0)
    ax.xaxis.set_major_formatter(
        charts.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if comp:
        # A key is needed here: many grey markers are otherwise unreadable.
        loc = (0.0, -0.1) if sq else (1.02, 1.0)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=loc,
            frameon=False,
            fontsize=11,
            labelcolor=INK,
            ncol=2 if sq else 1,
        )
    charts.save(fig, out)
