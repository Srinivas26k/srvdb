# SrvDB brand kit

The SrvDB mark is a cubist imperial eagle: the sharpest eye in nature,
built from flat planes. The planes stand for embeddings; the pupil sits on
the edge of the iris, the "point on the unit circle" that cosine search
compares.

## Files

| File | Use |
|------|-----|
| `srvdb-logo-horizontal.svg` | Default logo on light backgrounds (README, docs, slides) |
| `srvdb-logo-horizontal-dark.svg` | Logo on dark backgrounds (navy panel included) |
| `srvdb-eagle.svg` | Symbol with its cream disc: avatars, stickers, app icons |
| `srvdb-eagle-transparent.svg` | Symbol without the disc, for coloured backgrounds |
| `srvdb-eagle-small.svg` | Simplified symbol for 48 px and below (favicons) |
| `srvdb-eagle-mono-navy.svg` | One-colour version (print, embossing, single-ink) |
| `srvdb-eagle-mono-white.svg` | One-colour version reversed, for dark backgrounds |
| `srvdb-wordmark.svg` | Name only, for places the symbol already appears |
| `png/` | Raster exports: 512/1024 px symbol, lockups, `favicon.ico`, `apple-touch-icon.png`, `social-card.png` (GitHub social preview, 1280×640) |

## Colours

| Name | HEX | Role |
|------|-----|------|
| Imperial navy | `#1B2A4A` | Primary, wordmark |
| Ink | `#0F1A30` | Brow, dark backgrounds |
| Crimson | `#C8102E` | Accent |
| Gold | `#F2B134` | Accent, beak, highlights |
| Austerlitz cream | `#F6E7C8` | Disc, text on dark |

## Typography

The wordmark is set in **Cinzel Bold** (SIL Open Font License 1.1,
<https://github.com/NDISCOVER/Cinzel>) and converted to outlines, so no
font is needed to display it.

## Rules

- Keep clear space around the logo of at least the eye's diameter × 2.
- Below 48 px use `srvdb-eagle-small.svg`; below 24 px use the symbol alone.
- Do not recolour individual planes, add effects (shadows, gradients,
  outlines), stretch, rotate, or place the full-colour mark on busy images.
- Use the mono versions when only one colour is available.

## Rebuilding

`tools/build_kit.py` generates every SVG from code. Download the Cinzel
variable font (`Cinzel[wght].ttf`) to `tools/font/Cinzel.ttf`, then:

```bash
uv run --no-project --with fonttools python assets/brand/tools/build_kit.py assets/brand
```

The logo is © Srinivas Nampalli. The Apache-2.0 licence covers SrvDB's
code; it does not grant trademark rights in the SrvDB name or logo
(Apache-2.0 section 6).
