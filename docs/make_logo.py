#!/usr/bin/env python3
"""Generate the project logo.

    python docs/make_logo.py

The mark is a tire ring containing two of the project's guarantees, drawn from the
actual model rather than sketched:

* the **odd-symmetric force curve**, computed by
  :py:func:`tire_nn.physics.pacejka.pacejka_longitudinal`, passing exactly through the
  centre — zero force at zero slip;
* the **friction bound** it saturates into, dashed.

Note the axes: this is force against *slip*, so the bound is the pair of lines
``|F| = mu*Fz``. (The friction *ellipse* bounds ``Fx`` against ``Fy`` — a different
plane, and drawing an ellipse here would be a category error.)

Outputs (all committed):

===========================================  =========================================
``docs/source/_static/logo-mark.svg``        square mark, no text — docs sidebar
``docs/source/_static/logo.svg``             mark + wordmark, dark text — README (light)
``docs/source/_static/logo-dark.svg``        mark + wordmark, light text — dark backgrounds
``docs/source/_static/favicon.png``          64x64 raster
``docs/source/_static/logo-social.png``      1280x640 banner for the repository preview
===========================================  =========================================
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tire_nn.physics.pacejka import MFParams, pacejka_longitudinal  # noqa: E402

OUT = ROOT / "docs" / "source" / "_static"
OUT.mkdir(parents=True, exist_ok=True)

# Palette shared with the documentation diagrams: blue = fixed physics, orange = learned.
BLUE = "#2c5f9e"
BLUE_LIGHT = "#7aa7d4"
ORANGE = "#e8763a"
INK = "#1f2a37"
PAPER = "#ffffff"

CX = CY = 50.0
R_RING = 45.0
RX, RY = 32.0, 24.0          # half-width (slip axis) and the |F| = mu*Fz bound
KAPPA_MAX = 0.34


def force_curve(n: int = 120) -> list[tuple[float, float]]:
    """The real longitudinal force curve, mapped into the mark's coordinates.

    Uses ``mu = 1`` and ``k_mu = 0`` so the curve is bounded by exactly the drawn
    ellipse — the picture is then a true statement about the model, not an impression
    of one.
    """
    kappa = torch.linspace(-KAPPA_MAX, KAPPA_MAX, n)
    Fz = torch.full_like(kappa, 1000.0)
    p = MFParams(B=9.0, C=1.62, E=0.4, mu=1.0, k_mu=0.0)
    Fx, _ = pacejka_longitudinal(kappa, Fz, p)
    normalised = (Fx / (1.0 * Fz)).tolist()          # in [-1, 1]
    return [(CX + (float(k) / KAPPA_MAX) * RX, CY - f * RY)
            for k, f in zip(kappa.tolist(), normalised)]


def polyline(points, precision: int = 2) -> str:
    return " ".join(f"{x:.{precision}f},{y:.{precision}f}" for x, y in points)


def mark(ring_colour: str = BLUE) -> str:
    """The square mark, as SVG elements on a 0..100 canvas."""
    return f"""  <g>
    <!-- tire carcass -->
    <circle cx="{CX}" cy="{CY}" r="{R_RING}" fill="none"
            stroke="{ring_colour}" stroke-width="6"/>
    <!-- the friction bound |F| = mu*Fz, which the curve saturates into but never crosses -->
    <line x1="{CX - RX - 2}" y1="{CY - RY}" x2="{CX + RX + 2}" y2="{CY - RY}"
          stroke="{BLUE_LIGHT}" stroke-width="2.1" stroke-dasharray="5 4" stroke-linecap="round"/>
    <line x1="{CX - RX - 2}" y1="{CY + RY}" x2="{CX + RX + 2}" y2="{CY + RY}"
          stroke="{BLUE_LIGHT}" stroke-width="2.1" stroke-dasharray="5 4" stroke-linecap="round"/>
    <!-- axes through the origin -->
    <line x1="{CX - RX - 4}" y1="{CY}" x2="{CX + RX + 4}" y2="{CY}"
          stroke="{ring_colour}" stroke-width="1" opacity="0.35"/>
    <line x1="{CX}" y1="{CY - RY - 4}" x2="{CX}" y2="{CY + RY + 4}"
          stroke="{ring_colour}" stroke-width="1" opacity="0.35"/>
    <!-- the model's own force curve, odd and through the origin -->
    <polyline points="{polyline(force_curve())}" fill="none"
              stroke="{ORANGE}" stroke-width="5.2"
              stroke-linecap="round" stroke-linejoin="round"/>
    <!-- zero slip, zero force -->
    <circle cx="{CX}" cy="{CY}" r="3.4" fill="{ORANGE}"/>
    <circle cx="{CX}" cy="{CY}" r="1.5" fill="{PAPER}"/>
  </g>"""


def mark_simple(ring_colour: str = BLUE) -> str:
    """A stripped-down mark for very small sizes.

    At 32-64 px the axes and bound lines turn to mush and only muddy the silhouette, so
    the favicon keeps just the ring, a heavier curve and the origin dot.
    """
    return f"""  <g>
    <circle cx="{CX}" cy="{CY}" r="{R_RING - 2}" fill="none"
            stroke="{ring_colour}" stroke-width="9"/>
    <polyline points="{polyline(force_curve(80))}" fill="none"
              stroke="{ORANGE}" stroke-width="9.5"
              stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="{CX}" cy="{CY}" r="3.2" fill="{PAPER}"/>
  </g>"""


def write(path: Path, content: str) -> None:
    path.write_text(content)
    print(f"  {path.relative_to(ROOT)}")


def build_mark() -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"
     width="100" height="100" role="img"
     aria-label="tire_physics_nn: a tire with an odd-symmetric force curve inside its friction ellipse">
  <title>tire_physics_nn</title>
{mark()}
</svg>
"""
    write(OUT / "logo-mark.svg", svg)


def build_wordmark(name: str, text_colour: str, ring_colour: str, subtitle_opacity: float) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 110"
     width="460" height="110" role="img" aria-label="tire_physics_nn">
  <title>tire_physics_nn — physics-encoded neural tire models</title>
  <g transform="translate(5, 5)">
{mark(ring_colour)}
  </g>
  <text x="128" y="56" font-family="'DejaVu Sans Mono', 'SFMono-Regular', Menlo, monospace"
        font-size="30" font-weight="700" fill="{text_colour}">tire_physics_nn</text>
  <text x="130" y="80" font-family="'DejaVu Sans', Helvetica, Arial, sans-serif"
        font-size="14.5" fill="{text_colour}" opacity="{subtitle_opacity}">
    physics-encoded neural tire models
  </text>
</svg>
"""
    write(OUT / name, svg)


def build_rasters() -> None:
    try:
        import cairosvg
    except ImportError:
        print("  (cairosvg not installed — skipping PNG output; the SVGs are enough "
              "for the docs, but the favicon and social banner will not be regenerated)")
        return

    favicon_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"
     width="100" height="100" role="img" aria-label="tire_physics_nn">
{mark_simple()}
</svg>
"""
    write(OUT / "logo-favicon.svg", favicon_svg)
    for size in (32, 64):
        name = "favicon.png" if size == 64 else f"favicon-{size}.png"
        cairosvg.svg2png(url=str(OUT / "logo-favicon.svg"),
                         write_to=str(OUT / name),
                         output_width=size, output_height=size)
        print(f"  {(OUT / name).relative_to(ROOT)}")

    banner = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640">
  <rect width="1280" height="640" fill="{PAPER}"/>
  <g transform="translate(430, 120) scale(4.2)">
{mark()}
  </g>
  <text x="640" y="590" text-anchor="middle"
        font-family="'DejaVu Sans Mono', monospace" font-size="52" font-weight="700"
        fill="{INK}">tire_physics_nn</text>
</svg>
"""
    tmp = OUT / "_banner.svg"
    tmp.write_text(banner)
    cairosvg.svg2png(url=str(tmp), write_to=str(OUT / "logo-social.png"),
                     output_width=1280, output_height=640)
    tmp.unlink()
    print(f"  {(OUT / 'logo-social.png').relative_to(ROOT)}")


def main() -> None:
    print("Generating logo assets:")
    build_mark()
    build_wordmark("logo.svg", INK, BLUE, 0.75)
    build_wordmark("logo-dark.svg", "#f2f5f8", "#8fb8e0", 0.8)
    build_rasters()
    print("done.")


if __name__ == "__main__":
    main()
