#!/usr/bin/env python3
"""
Gauge icon generator (0..1), red=low on left, green=high on right.

Example:
    python ~/code/shitspotter/papers/wacv_2026/presentation/make_gague_image.py --score 0.7 --out gauge_07.png --label AP
    python ~/code/shitspotter/papers/wacv_2026/presentation/make_gague_image.py --score 0.2 --out gauge_02.png --label AP
"""

import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyBboxPatch, Polygon


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def draw_gauge(
    score: float,
    out_path: str,
    label: str = "AP",
    dpi: int = 200,
    facecolor: str = "white",
    value_fmt: str = "{:.1f}",
):
    score = clamp(float(score), 0.0, 1.0)

    # Geometry
    r_outer = 1.00
    r_inner = 0.60
    ring_width = r_outer - r_inner

    # Colors: LOW=red (left), MID=yellow (top), HIGH=green (right)
    c_red = "#d7191c"
    c_yel = "#fddc3a"
    c_grn = "#1a9641"

    # Figure
    fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
    fig.patch.set_facecolor(facecolor)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Gauge ring (top semicircle) ---
    segments = [
        (120, 180, c_red),  # left (low)
        (60, 120, c_yel),   # mid
        (0, 60, c_grn),     # right (high)
    ]
    for a1, a2, col in segments:
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=r_outer,
                theta1=a1,
                theta2=a2,
                width=ring_width,
                facecolor=col,
                edgecolor="none",
                zorder=2,
            )
        )

    # Inner “dial” circle (draw BEFORE the base; base will mask the bottom)
    ax.add_patch(
        Circle((0, 0), r_inner - 0.02, facecolor="white", edgecolor="none", zorder=3)
    )

    # Ticks
    tick_angles = [180, 150, 120, 90, 60, 30, 0]
    for ang in tick_angles:
        t = math.radians(ang)
        x1, y1 = 0.90 * math.cos(t), 0.90 * math.sin(t)
        x2, y2 = 0.78 * math.cos(t), 0.78 * math.sin(t)
        ax.plot([x1, x2], [y1, y2], color="white", lw=3, solid_capstyle="round", zorder=4)

    # Needle: score=0 -> left (180°), score=1 -> right (0°)
    theta_deg = 180.0 * (1.0 - score)
    theta = math.radians(theta_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    px, py = -dy, dx  # perpendicular

    tip_len = 0.92
    base_len = 0.20
    half_width = 0.045

    tip = (tip_len * dx, tip_len * dy)
    left_base = (base_len * dx + half_width * px, base_len * dy + half_width * py)
    right_base = (base_len * dx - half_width * px, base_len * dy - half_width * py)

    ax.add_patch(
        Polygon([left_base, tip, right_base], closed=True, facecolor="#1b1f23", edgecolor="none", zorder=6)
    )
    ax.add_patch(Circle((0, 0), 0.12, facecolor="#2a2f35", edgecolor="none", zorder=70))
    ax.add_patch(Circle((0, 0), 0.075, facecolor="#111418", edgecolor="none", zorder=80))

    # --- Base (draw AFTER the dial so it masks the lower part of the circle) ---
    base_top = -0.05
    base_height = 0.40
    base = FancyBboxPatch(
        (-1.05, base_top - base_height),
        2.10,
        base_height,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0,
        facecolor="#111418",
        zorder=10,
    )
    ax.add_patch(base)

    # Base text (on top of base)
    ax.text(
        0,
        base_top - base_height / 2,
        f"{label}: {value_fmt.format(score)}",
        ha="center",
        va="center",
        fontsize=34,
        fontweight="bold",
        color="white",
        zorder=11,
    )

    # Outer labels (draw last so they sit above the base edge)
    def place_text(txt, ang_deg, r, **kw):
        t = math.radians(ang_deg)
        ax.text(r * math.cos(t), r * math.sin(t), txt, ha="center", va="center", **kw)

    place_text("0", 175, 0.82, fontsize=18, fontweight="bold", color="#111418", zorder=12)
    place_text("0.5", 90, 0.88, fontsize=18, fontweight="bold", color="#111418", zorder=12)
    place_text("1", 5, 0.82, fontsize=18, fontweight="bold", color="#111418", zorder=12)

    # Frame
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.90, 1.15)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor=facecolor)
    import kwplot
    kwplot.cropwhite_ondisk(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score", type=float, required=True, help="Score in [0,1]")
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    p.add_argument("--label", type=str, default="AP", help="Label prefix (e.g. AP)")
    p.add_argument("--dpi", type=int, default=200, help="Output DPI")
    p.add_argument("--value-fmt", type=str, default="{:.1f}", help="Python format string, e.g. '{:.2f}'")
    args = p.parse_args()

    draw_gauge(
        score=args.score,
        out_path=args.out,
        label=args.label,
        dpi=args.dpi,
        value_fmt=args.value_fmt,
    )


if __name__ == "__main__":
    main()
