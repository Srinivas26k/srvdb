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
            f"{sub} · recall@10 {modes[k]['recall_at_10']:.3f}",
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
            sub += f" · no rerank {nr['recall_at_10']:.3f}"
        rows.append(
            Row(
                lbl,
                sub,
                modes[k]["recall_at_10"],
                f"{modes[k]['recall_at_10']:.3f}",
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
    import matplotlib.image as mpimg

    modes = ds["modes"]
    k = hero_key(modes)
    lbl = dict((a, b) for a, b, _ in ROWS)[k]
    ratio = code_bytes(modes["flat"]) / code_bytes(modes[k])
    fig, top = charts.figure(size, "", "", footer(env, name, ds))
    sq = size == "square"
    ax = fig.add_axes(
        (0.05, 0.62 if sq else 0.5, 0.25 if sq else 0.2, 0.3 if sq else 0.4)
    )
    ax.imshow(mpimg.imread(EAGLE))
    ax.axis("off")
    x = 0.05 if sq else 0.3
    fig.text(
        x,
        0.52 if sq else 0.82,
        f"{lbl}: {ratio:.0f}× smaller search codes",
        fontsize=40 if sq else 44,
        fontweight="bold",
        color=ACCENT,
        va="top",
    )
    fig.text(
        x,
        0.44 if sq else 0.68,
        f"recall@10 {modes[k]['recall_at_10']:.3f} on {NICE[name]} "
        f"({modes[k]['dim']:,}-d, {modes[k]['n']:,} vectors)",
        fontsize=24,
        color=INK,
        va="top",
    )
    fig.text(
        x,
        0.38 if sq else 0.6,
        f"{code_bytes(modes['flat']):,.0f} B → "
        f"{code_bytes(modes[k]):,.0f} B per vector scanned; "
        "full float32 vectors stay on disk for exact rerank",
        fontsize=15,
        color=GREY,
        va="top",
    )
    crash = env.get("_crash")
    if crash and crash.get("passed"):
        fig.text(
            x,
            0.32 if sq else 0.53,
            f"Committed records lost after {crash['kills_total']:,} kill -9s: 0",
            fontsize=18,
            color=INK,
            va="top",
        )
    fig.text(
        x,
        0.22 if sq else 0.36,
        "SrvDB · embedded vector database · Rust + Python",
        fontsize=20,
        color=INK,
        fontweight="bold",
        va="top",
    )
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
