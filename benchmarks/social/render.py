"""Render the social PNGs from results.json only (no measuring here).

    python benchmarks/social/render.py

Writes benchmarks/social/img/<chart>-<landscape|square>.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import charts  # noqa: E402
import compare_chart  # noqa: E402
from charts import ACCENT, GREY, INK, Row  # noqa: E402

ROOT = HERE.parent.parent
IMG = HERE / "img"
EAGLE = ROOT / "assets/brand/png/srvdb-eagle-512.png"
HEADLINE = "dbpedia-openai-100k"
ROWS = [
    ("flat", "float32", "exact scan"),
    ("sq8", "SQ8", "8-bit scalar"),
    ("pq", "PQ", "product quantization"),
    ("rabitq4", "RaBitQ 4-bit", "rerank on"),
    ("rabitq2", "RaBitQ 2-bit", "rerank on"),
    ("rabitq1", "RaBitQ 1-bit", "rerank on"),
]
NICE = {
    "dbpedia-openai-100k": "DBpedia OpenAI ada-002",
    "nytimes-256-angular": "NYTimes",
    "glove-100-angular": "GloVe-100",
    "synthetic-768": "synthetic clustered",
}
MIN_RECALL = 0.9


def footer(env: dict, name: str, ds: dict) -> str:
    m = next(iter(ds["modes"].values()))
    return (
        f"SrvDB {env['srvdb_version']} · {NICE[name]} · dim {m['dim']} · "
        f"n={m['n']:,} · {m['queries']:,} queries · {env['cpu']} · "
        f"commit {env['git_commit'][:7]} · github.com/Srinivas26k/srvdb"
    )


def rect(size: str, band: tuple, left: float) -> tuple:
    top, bottom = band
    left = left + (0.08 if size == "square" else 0)
    return (left, bottom, 0.95 - left, top - bottom)


def code_bytes(m: dict) -> float:
    """Measured search-code bytes per vector: memory_report vector_bytes / n."""
    return m["memory_report"]["vector_bytes"] / m["n"]


def hero_key(modes: dict) -> str:
    ok = [k for k, *_ in ROWS if k in modes and modes[k]["recall_at_10"] >= MIN_RECALL]
    return min(ok, key=lambda k: code_bytes(modes[k]))


def memory(env, name, ds, size, out):
    modes, hero = ds["modes"], hero_key(ds["modes"])
    rows = [
        Row(
            lbl,
            f"{sub} · recall@10 {modes[k]['recall_at_10']:.4f}",
            code_bytes(modes[k]),
            f"{code_bytes(modes[k]):,.0f} B",
            k == hero,
        )
        for k, lbl, sub in ROWS
        if k in modes
    ]
    rows.sort(key=lambda r: -r.value)
    dim = modes["flat"]["dim"]
    fig, top = charts.figure(
        size,
        f"Bytes per vector searched: {NICE[name]}, {dim}-d",
        "Bytes per vector the search scans, from db.memory_report() "
        "(lower is better). RaBitQ also keeps float32 on disk for rerank.",
        footer(env, name, ds),
    )
    charts.bars(fig, rect(size, top, 0.25), rows, rows[0].value, arrows=True)
    charts.save(fig, out)


def recall(env, name, ds, size, out):
    modes, hero = ds["modes"], hero_key(ds["modes"])
    flat = code_bytes(modes["flat"])
    rows = []
    for k, lbl, _ in ROWS:
        if k not in modes:
            continue
        b = code_bytes(modes[k])
        nr = modes.get(f"{k}-norerank")
        sub = f"{b:,.0f} B/vector · " + (
            "baseline" if k == "flat" else f"{flat / b:.1f}× smaller"
        )
        if nr:
            sub += f" · no rerank {nr['recall_at_10']:.4f}"
        rows.append(
            Row(
                lbl,
                sub,
                modes[k]["recall_at_10"],
                f"{modes[k]['recall_at_10']:.4f}",
                k == hero,
            )
        )
    fig, top = charts.figure(
        size,
        f"Recall@10 by mode: {NICE[name]}",
        "Recall@10 against exact ground truth, default settings (higher is better)",
        footer(env, name, ds),
    )
    charts.bars(fig, rect(size, top, 0.31), rows, 1.0, arrows=False)
    charts.save(fig, out)


def hero_card(env, name, ds, size, out):
    import textwrap

    import matplotlib.image as mpimg

    modes = ds["modes"]
    k = hero_key(modes)
    m = modes[k]
    lbl = dict((a, b) for a, b, _ in ROWS)[k]
    ratio = code_bytes(modes["flat"]) / code_bytes(m)
    fig, _ = charts.figure(size, "", "", footer(env, name, ds))
    sq = size == "square"
    w, h = charts.SIZES[size]
    ax = fig.add_axes((0.05, 0.66, 0.26, 0.26) if sq else (0.04, 0.42, 0.22, 0.4))
    ax.imshow(mpimg.imread(EAGLE))
    ax.axis("off")
    x, y = (0.06, 0.6) if sq else (0.3, 0.85)
    width = 0.88 if sq else 0.66  # usable fraction of the figure width
    crash = env.get("_crash")
    lines = [
        (f"{lbl}: {ratio:.0f}× smaller search codes", 46, ACCENT, "bold", 0.62),
        (
            f"recall@10 {m['recall_at_10']:.4f} on {NICE[name]} embeddings "
            f"({m['dim']:,}-d, {m['n']:,} vectors)",
            24,
            INK,
            "normal",
            0.55,
        ),
        (
            f"{code_bytes(modes['flat']):,.0f} B → {code_bytes(m):,.0f} B per vector "
            "scanned; float32 vectors stay on disk for exact rerank",
            16,
            GREY,
            "normal",
            0.55,
        ),
    ]
    if crash and crash.get("passed"):
        lines.append(
            (
                f"Committed records lost after {crash['kills_total']:,} kill -9s: 0",
                20,
                INK,
                "normal",
                0.55,
            )
        )
    lines.append(
        ("SrvDB · embedded vector database · Rust + Python", 20, INK, "bold", 0.6)
    )
    for text, fs, color, weight, cw in lines:
        # cw: average glyph width as a fraction of the font size.
        chars = int(width * w * 72 / (fs * cw))
        text = textwrap.fill(text, chars)
        fig.text(
            x,
            y,
            text,
            fontsize=fs,
            color=color,
            fontweight=weight,
            va="top",
            linespacing=1.2,
        )
        y -= (fs * 1.25 * (text.count("\n") + 1) / 72 + 0.3) / h
    charts.save(fig, out)


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    env = res["environment"] | {"_crash": res.get("crash_test")}
    IMG.mkdir(exist_ok=True)
    for name, ds in res["datasets"].items():
        if "flat" not in ds["modes"]:
            continue
        for size in charts.SIZES:
            memory(env, name, ds, size, IMG / f"memory-{name}-{size}.png")
            recall(env, name, ds, size, IMG / f"recall-{name}-{size}.png")
            if ds.get("competitors"):
                compare_chart.draw(
                    env,
                    name,
                    NICE[name],
                    ds,
                    size,
                    footer(env, name, ds),
                    IMG / f"compare-{name}-{size}.png",
                )
            if name == HEADLINE:
                memory(env, name, ds, size, IMG / f"memory-{size}.png")
                recall(env, name, ds, size, IMG / f"recall_vs_memory-{size}.png")
                hero_card(env, name, ds, size, IMG / f"hero-{size}.png")
    print(f"wrote {len(list(IMG.glob('*.png')))} PNGs to {IMG}", file=sys.stderr)


if __name__ == "__main__":
    main()
