"""Drawing primitives for the social charts: labelled horizontal bars.

Style: white background, large top-left title, one bar per row with a big
label and a grey sub-label, grey bars except one accent hero row, value
right of each bar, curved "N x smaller" arrows, faint vertical grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ACCENT = "#5B6CF0"
BAR = "#C9CCD3"
INK = "#16181D"
GREY = "#6B7080"
SIZES = {"landscape": (16, 9), "square": (12, 12)}  # inches at 100 dpi
SAVE_DPI = 200  # 2x: 1600x900 -> 3200x1800 pixels


def font() -> str:
    # Only families with a static bold face: matplotlib cannot pick weights
    # from variable fonts (Ubuntu ships as one on recent Ubuntu releases).
    bold = {f.name for f in fm.fontManager.ttflist if f.weight in (700, "bold")}
    return next(
        (n for n in ("Ubuntu", "Inter", "DejaVu Sans") if n in bold), "sans-serif"
    )


plt.rcParams.update({"font.family": font(), "axes.edgecolor": INK})


@dataclass
class Row:
    label: str
    sub: str
    value: float
    text: str
    hero: bool = False


def figure(size: str, title: str, subtitle: str, footer: str):
    """White figure with wrapped title, subtitle and footer; returns the
    figure and the free band (bottom, top) in figure coordinates."""
    import textwrap

    w, h = SIZES[size]
    fig = plt.figure(figsize=(w, h), dpi=100, facecolor="white")
    title = textwrap.fill(title, int(w * 3.0))
    subtitle = textwrap.fill(subtitle, int(w * 9.5))
    footer = textwrap.fill(footer, int(w * 13))
    y = 1 - 0.5 / h
    fig.text(
        0.045,
        y,
        title,
        fontsize=32,
        fontweight="bold",
        color=INK,
        va="top",
        linespacing=1.15,
    )
    y -= (0.56 * (title.count("\n") + 1) + 0.12) / h
    fig.text(0.045, y, subtitle, fontsize=16, color=GREY, va="top")
    y -= (0.3 * (subtitle.count("\n") + 1) + 0.35) / h
    fig.text(0.045, 0.3 / h, footer, fontsize=11, color=GREY, va="bottom")
    bottom = (0.3 + 0.2 * (footer.count("\n") + 1) + 0.45) / h
    top = (y, bottom)
    return fig, top


def bars(
    fig,
    rect,
    rows: list[Row],
    unit_max: float,
    arrows: bool,
    ratio_fmt: str = "{:.1f}× smaller",
) -> None:
    """Draw labelled rows in rect=(left, bottom, width, height)."""
    ax = fig.add_axes(rect)
    n = len(rows)
    ys = list(range(n))[::-1]
    ax.set_xlim(0, unit_max * 1.32)
    ax.set_ylim(-0.6, n - 0.4)
    ends = []
    for y, r in zip(ys, rows):
        ax.barh(y, r.value, height=0.56, color=ACCENT if r.hero else BAR, zorder=2)
        ends.append(
            ax.text(
                r.value + unit_max * 0.012,
                y,
                r.text,
                va="center",
                fontsize=19,
                color=ACCENT if r.hero else INK,
                fontweight="bold" if r.hero else "normal",
                zorder=3,
            )
        )
        ax.text(
            -unit_max * 0.02,
            y + 0.08,
            r.label,
            ha="right",
            va="bottom",
            fontsize=20,
            color=ACCENT if r.hero else INK,
            fontweight="bold" if r.hero else "normal",
        )
        ax.text(
            -unit_max * 0.02,
            y - 0.05,
            r.sub,
            ha="right",
            va="top",
            fontsize=12.5,
            color=GREY,
        )
    if arrows:
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        right = [inv.transform(t.get_window_extent(renderer).p1)[0] for t in ends]
        for i, (a, b) in enumerate(zip(rows, rows[1:])):
            if b.value <= 0 or a.value <= b.value:
                continue
            y0, y1 = ys[i], ys[i + 1]
            # From just right of row a's value text down to row b's.
            xs, xe = right[i] + unit_max * 0.012, right[i + 1] + unit_max * 0.012
            # Near-vertical hops bow out; long horizontal hops stay flat.
            wide = abs(xs - xe) > unit_max * 0.15
            rad = (0.12 if wide else 0.5) * (-1 if xs > xe else 1)
            ax.add_patch(
                FancyArrowPatch(
                    (xs, y0),
                    (xe, y1),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>",
                    mutation_scale=16,
                    lw=1.4,
                    color=INK,
                    zorder=4,
                    shrinkA=2,
                    shrinkB=2,
                )
            )
            lx = (
                (xs + xe) / 2 + unit_max * 0.03
                if wide
                else max(xs, xe) + unit_max * 0.07
            )
            ax.text(
                lx,
                (y0 + y1) / 2,
                ratio_fmt.format(a.value / b.value),
                va="center",
                fontsize=13,
                color=INK,
            )
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=GREY, labelsize=12, length=0)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if unit_max > 10 else f"{v:.1f}"
        )
    )
    if unit_max <= 1:
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", color="#ECEDF1", lw=1, zorder=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)


def save(fig, path) -> None:
    fig.savefig(path, dpi=SAVE_DPI, facecolor="white")
    plt.close(fig)
