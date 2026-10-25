"""Build the SrvDB eagle logo kit into OUT (argv[1])."""
import sys, os, re
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen

OUT = sys.argv[1]; HERE = os.path.dirname(__file__)
os.makedirs(OUT, exist_ok=True)
NAVY, NAVY2, NAVY3, INK = "#1B2A4A", "#2E4F86", "#24406E", "#0F1A30"
CRIMSON, CRIMSON2 = "#C8102E", "#9E0B24"
GOLD, GOLD2, GOLD3, ORANGE = "#F2B134", "#E8A317", "#FFD166", "#E07A10"
CREAM = "#F6E7C8"

def svg(body, w=256, h=256, title="SrvDB"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{title}"><title>{title}</title>{body}</svg>\n')

def eagle(disc=True, mono=None):
    """Full eagle. mono: one colour for every plane (planes split by disc-coloured seams)."""
    c = (lambda col: mono) if mono else (lambda col: col)
    seam = f' stroke="{CREAM if disc else "#FFFFFF"}" stroke-width="4" stroke-linejoin="round"' if mono else ""
    p = lambda pts, col: f'<polygon points="{pts}" fill="{c(col)}"{seam}/>'
    b = f'<circle cx="128" cy="128" r="112" fill="{CREAM}"/>' if disc else ""
    b += p("40,228 52,186 66,118 126,154 122,228 104,206 92,232 74,204 58,230", NAVY3)
    b += p("126,154 172,144 152,170 140,232 122,228", GOLD2)
    b += p("66,118 106,70 140,62 122,108", NAVY)
    b += p("140,62 182,58 150,100 122,108", NAVY2)
    b += p("66,118 122,108 150,100 140,134 126,154", CRIMSON)
    b += p("150,100 186,100 192,132 172,144 126,154 140,134", CRIMSON2)
    b += p("190,132 216,140 222,160 198,150", ORANGE)
    b += p("186,100 212,84 238,94 252,122 248,152 230,180 226,156 216,140 190,132", GOLD)
    b += p("212,84 238,94 252,122 222,112", GOLD3)
    iris = CREAM if mono else GOLD
    b += f'<circle cx="162" cy="114" r="15" fill="{iris}"/><circle cx="171" cy="108" r="7" fill="{INK if not mono else mono}"/>'
    b += p("136,108 180,56 218,82 190,100 166,98", INK)
    return b

def eagle_small():
    """Small-size cut (favicons, <= 32 px): head only, four planes, scaled to fill."""
    b = f'<circle cx="128" cy="128" r="124" fill="{CREAM}"/>'
    b += '<g transform="translate(-38,-22) scale(1.1)">'
    b += f'<polygon points="72,196 66,118 140,62 182,58 150,100 186,100 192,132 126,154 124,208" fill="{NAVY}"/>'
    b += f'<polygon points="66,118 150,100 186,100 192,132 126,154" fill="{CRIMSON}"/>'
    b += f'<polygon points="186,100 212,84 238,94 252,122 248,152 230,180 226,156 216,140 190,132" fill="{GOLD}"/>'
    b += f'<circle cx="164" cy="114" r="17" fill="{GOLD}"/><circle cx="173" cy="107" r="8" fill="{INK}"/>'
    b += f'<polygon points="136,108 180,56 218,82 190,100 166,98" fill="{INK}"/>'
    return b + "</g>"

# Wordmark: "SrvDB" in Cinzel Bold (OFL), outlined
font = instancer.instantiateVariableFont(TTFont(os.path.join(HERE, "font", "Cinzel.ttf")), {"wght": 700})
gs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
upm = font["head"].unitsPerEm
def wordmark(text, height, color, tracking=0.06):
    """Return (svg path group, width) for text whose cap height is `height`."""
    cap = font["OS/2"].sCapHeight; s = height / cap
    x, paths = 0.0, []
    for ch in text:
        g = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, x, height)))
        paths.append(pen.getCommands())
        x += hmtx[g][0] * s + tracking * upm * s
    x -= tracking * upm * s
    return f'<path fill="{color}" d="{" ".join(paths)}"/>', x

def lockup(color_word, dark=False):
    word, ww = wordmark("SrvDB", 112, color_word)
    gap, h = 56, 256
    w = 256 + gap + ww + 24
    bg = f'<rect width="{w:.0f}" height="{h}" rx="32" fill="{INK}"/>' if dark else ""
    return svg(bg + eagle() + f'<g transform="translate({256 + gap},{(h - 112) / 2})">{word}</g>', w=round(w), h=h)

files = {
    "srvdb-eagle.svg": svg(eagle()),
    "srvdb-eagle-transparent.svg": svg(eagle(disc=False)),
    "srvdb-eagle-small.svg": svg(eagle_small()),
    "srvdb-eagle-mono-navy.svg": svg(eagle(disc=False, mono=NAVY)),
    "srvdb-eagle-mono-white.svg": svg(eagle(disc=False, mono="#FFFFFF").replace('stroke="#FFFFFF"', f'stroke="{INK}"')),
    "srvdb-logo-horizontal.svg": lockup(NAVY),
    "srvdb-logo-horizontal-dark.svg": lockup(CREAM, dark=True),
}
w, ww = wordmark("SrvDB", 112, NAVY)
files["srvdb-wordmark.svg"] = svg(w, w=round(ww) + 2, h=112)
for name, data in files.items():
    open(os.path.join(OUT, name), "w").write(data)
print("\n".join(files))
