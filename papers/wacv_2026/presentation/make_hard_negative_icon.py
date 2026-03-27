#!/usr/bin/env python3
"""
Hard-negative icon generator (transparent background) with modular drawing functions.

Fixes requested:
- Separate functions for each element (border, placeholder, magnifier, warning).
- Placeholder: big mountain on LEFT, sun scooted LEFT, overall placeholder less wide.
- Warning: exclamation centered in triangle (custom-drawn, not font-baseline dependent).
- Magnifier: handle aligned with radial direction at attachment point (=> perpendicular to tangent).
- Magnifier: question mark centered in the lens using real font glyph, with robust bbox-centering.

Requires: Pillow
  pip install pillow
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import math

RGBA = Tuple[int, int, int, int]


def load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Use a real font glyph for "?" if available
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default()


def center_text(draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str, font, fill: RGBA):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((cx - w / 2, cy - h / 2), text, font=font, fill=fill)


def dashed_rect(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, dash, gap, outline: RGBA, width: int):
    # top/bottom
    x = x0
    while x < x1:
        x2 = min(x + dash, x1)
        draw.line([(x, y0), (x2, y0)], fill=outline, width=width)
        draw.line([(x, y1), (x2, y1)], fill=outline, width=width)
        x += dash + gap
    # left/right
    y = y0
    while y < y1:
        y2 = min(y + dash, y1)
        draw.line([(x0, y), (x0, y2)], fill=outline, width=width)
        draw.line([(x1, y), (x1, y2)], fill=outline, width=width)
        y += dash + gap


def rotate_layer(layer: Image.Image, degrees: float, center_xy: Tuple[float, float]) -> Image.Image:
    return layer.rotate(degrees, resample=Image.Resampling.BICUBIC, center=center_xy)


@dataclass
class IconStyle:
    # Default: all black (note: on black viewer backgrounds you'll barely see it)
    border: RGBA = (0, 0, 0, 255)
    stroke: RGBA = (0, 0, 0, 255)
    fill: RGBA = (0, 0, 0, 255)

    placeholder_stroke: RGBA = (0, 0, 0, 170)
    placeholder_fill: RGBA = (0, 0, 0, 110)

    magnifier: Optional[RGBA] = None  # None -> stroke
    warning: Optional[RGBA] = None    # None -> fill

    # Stroke widths (scaled by canvas anyway)
    border_w_frac: float = 0.020
    placeholder_w_frac: float = 0.006
    magnifier_ring_w_frac: float = 0.030


@dataclass
class Layout:
    # Positions are fractions of the (supersampled) canvas
    pad_frac: float = 0.08
    corner_radius_frac: float = 0.08

    # Placeholder bbox (less wide than before)
    ph_x0: float = 0.16
    ph_y0: float = 0.22
    ph_x1: float = 0.46  # narrower than 0.50
    ph_y1: float = 0.46

    # Magnifier
    lens_cx: float = 0.43
    lens_cy: float = 0.34
    lens_r: float = 0.155
    handle_attach_angle_deg: float = 35.0  # down-right; radial direction => perpendicular to tangent
    handle_len: float = 0.23
    handle_w: float = 0.070

    # Warning triangle
    tri_cx: float = 0.60
    tri_cy: float = 0.64
    tri_w: float = 0.26
    tri_h: float = 0.22


def draw_border(draw: ImageDraw.ImageDraw, S: int, style: IconStyle, layout: Layout):
    pad = int(S * layout.pad_frac)
    r = int(S * layout.corner_radius_frac)
    w = max(2, int(S * style.border_w_frac))
    draw.rounded_rectangle([pad, pad, S - pad, S - pad], radius=r, outline=style.border, width=w)


def draw_placeholder(draw: ImageDraw.ImageDraw, S: int, style: IconStyle, layout: Layout):
    x0 = int(S * layout.ph_x0)
    y0 = int(S * layout.ph_y0)
    x1 = int(S * layout.ph_x1)
    y1 = int(S * layout.ph_y1)

    dash = int(S * 0.020)
    gap = int(S * 0.012)
    w = max(1, int(S * style.placeholder_w_frac))
    dashed_rect(draw, x0, y0, x1, y1, dash=dash, gap=gap, outline=style.placeholder_stroke, width=w)

    # Mountains: big on LEFT, small on RIGHT
    # Build a simple silhouette with two peaks
    base_y = int(y1 - (y1 - y0) * 0.12)
    left_base_x = int(x0 + (x1 - x0) * 0.10)
    right_base_x = int(x1 - (x1 - x0) * 0.10)

    big_peak_x = int(x0 + (x1 - x0) * 0.32)   # LEFT
    big_peak_y = int(y0 + (y1 - y0) * 0.45)

    small_peak_x = int(x0 + (x1 - x0) * 0.62)  # RIGHT
    small_peak_y = int(y0 + (y1 - y0) * 0.55)

    # Polygon: left base -> big peak -> valley -> small peak -> right base
    valley_x = int(x0 + (x1 - x0) * 0.48)
    valley_y = int(y0 + (y1 - y0) * 0.72)

    pts = [
        (left_base_x, base_y),
        (big_peak_x, big_peak_y),
        (valley_x, valley_y),
        (small_peak_x, small_peak_y),
        (right_base_x, base_y),
    ]
    draw.polygon(pts, fill=style.placeholder_fill)

    # Sun: scoot LEFT a bit relative to previous
    sun_r = int(S * 0.030)
    sun_cx = int(x0 + (x1 - x0) * 0.78)  # was ~0.85-ish; moved left
    sun_cy = int(y0 + (y1 - y0) * 0.18)
    draw.ellipse([sun_cx - sun_r, sun_cy - sun_r, sun_cx + sun_r, sun_cy + sun_r], fill=style.placeholder_fill)


def render_glyph_tight(
    text: str,
    font: ImageFont.ImageFont,
    fill=(0, 0, 0, 255),
    pad: int = 2,
) -> Image.Image:
    """
    Renders `text` to an RGBA image cropped to the actual alpha content (tight),
    with optional padding.
    """
    # Make a generous temporary canvas
    tmp = Image.new("RGBA", (font.size * 4, font.size * 4), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)

    # Draw near the middle to avoid clipping; use an offset baseline.
    # (We draw once, then crop by alpha.)
    x = tmp.size[0] // 2
    y = tmp.size[1] // 2
    td.text((x, y), text, font=font, fill=fill, anchor="mm")  # middle-middle anchor

    # Crop to non-transparent pixels (alpha channel)
    alpha = tmp.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        # Nothing rendered; return a 1x1 transparent image
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    glyph = tmp.crop(bbox)

    if pad > 0:
        out = Image.new("RGBA", (glyph.size[0] + 2 * pad, glyph.size[1] + 2 * pad), (0, 0, 0, 0))
        out.alpha_composite(glyph, (pad, pad))
        return out

    return glyph


def paste_centered(
    base: Image.Image,
    sprite: Image.Image,
    cx: float,
    cy: float,
) -> None:
    """
    Alpha-composites `sprite` onto `base` so that sprite's pixel bounds are centered at (cx, cy).
    """
    x = int(round(cx - sprite.size[0] / 2))
    y = int(round(cy - sprite.size[1] / 2))
    base.alpha_composite(sprite, (x, y))


def draw_magnifier(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    S: int,
    style: IconStyle,
    layout: Layout,
):
    mag_col = style.magnifier if style.magnifier is not None else style.stroke

    cx = S * layout.lens_cx
    cy = S * layout.lens_cy
    r = S * layout.lens_r
    ring_w = max(2, int(S * style.magnifier_ring_w_frac))

    # Ring
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=mag_col, width=ring_w)

    # Handle: align with radial direction at attachment point (=> perpendicular to circle tangent)
    theta = math.radians(layout.handle_attach_angle_deg)
    attach_x = cx + r * math.cos(theta)
    attach_y = cy + r * math.sin(theta)

    handle_len = S * layout.handle_len
    handle_w = S * layout.handle_w

    # Draw handle axis-aligned to +x starting at attachment, then rotate around attachment
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)

    hx0 = attach_x
    hy0 = attach_y - handle_w / 2
    hx1 = attach_x + handle_len
    hy1 = attach_y + handle_w / 2
    ld.rounded_rectangle([hx0, hy0, hx1, hy1], radius=handle_w / 2, fill=mag_col)

    # Rotate by theta degrees around attachment point
    layer = rotate_layer(layer, degrees=-layout.handle_attach_angle_deg, center_xy=(attach_x, attach_y))
    img.alpha_composite(layer)

    # Real question mark: center using text bbox
    font = load_font(int(S * 0.24))

    glyph_img = render_glyph_tight("!", font=font, fill=mag_col, pad=max(1, int(S * 0.002)))
    paste_centered(img, glyph_img, cx, cy)

    # # Slight nudge can be applied if you want; keep zero for true bbox center
    # qx, qy = cx, cy
    # center_text(draw, qx, qy, "?", font=font, fill=mag_col)


def draw_centered_exclamation(draw: ImageDraw.ImageDraw, cx: float, cy: float, scale: float, color: RGBA):
    """
    Custom-drawn exclamation mark centered at (cx, cy) in a way that's visually centered.
    """
    bar_h = scale * 0.62
    bar_w = max(1.0, scale * 0.14)
    dot_r = max(1.0, scale * 0.10)
    gap = scale * 0.08

    # Bar centered slightly above the center because dot adds visual weight below
    bar_cy = cy - (dot_r + gap) * 0.35
    x0 = cx - bar_w / 2
    y0 = bar_cy - bar_h / 2
    x1 = cx + bar_w / 2
    y1 = bar_cy + bar_h / 2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=bar_w / 2, fill=color)

    # Dot
    dot_cy = y1 + gap + dot_r
    draw.ellipse([cx - dot_r, dot_cy - dot_r, cx + dot_r, dot_cy + dot_r], fill=color)


def draw_warning(draw: ImageDraw.ImageDraw, S: int, style: IconStyle, layout: Layout):
    warn_fill = style.warning if style.warning is not None else style.fill

    cx = S * layout.tri_cx
    cy = S * layout.tri_cy
    w = S * layout.tri_w
    h = S * layout.tri_h

    p1 = (cx, cy - h / 2)
    p2 = (cx - w / 2, cy + h / 2)
    p3 = (cx + w / 2, cy + h / 2)
    draw.polygon([p1, p2, p3], fill=warn_fill)

    # Place exclamation at triangle's visual center:
    # For an isosceles triangle, centroid is 1/3 from base toward apex.
    centroid_x = cx
    centroid_y = (p1[1] + p2[1] + p3[1]) / 3.0
    # Slight upward nudge often looks better in warning triangles
    centroid_y -= h * 0.02

    # FIXME USE render_glyph_tight

    draw_centered_exclamation(
        draw,
        centroid_x,
        centroid_y,
        scale=min(w, h) * 0.62,
        color=(255, 255, 255, 255),
    )


def make_icon(
    out_path: str = "hard_negative_icon.png",
    size: int = 1024,
    supersample: int = 4,
    style: IconStyle = IconStyle(),
    layout: Layout = Layout(),
    include_warning: bool = True,
):
    S = size * supersample
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    draw_border(d, S * 0.7, style, layout)
    draw_placeholder(d, S, style, layout)
    draw_magnifier(img, d, S, style, layout)

    if include_warning:
        draw_warning(d, S, style, layout)

    if supersample != 1:
        img = img.resize((size, size), resample=Image.Resampling.LANCZOS)

    img.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Default: all black.
    # NOTE: if you view the transparent PNG on a black background, black strokes won't show.
    make_icon(
        out_path="hard_negative_icon.png",
        size=1024,
        supersample=4,
        include_warning=False,
        style=IconStyle(
            # If you want to *see* it on dark backgrounds, switch border/strokes to white:
            # border=(255,255,255,255), stroke=(255,255,255,255),
            # placeholder_stroke=(255,255,255,170), placeholder_fill=(255,255,255,110),
            # magnifier=(255,255,255,255),
            # warning=(255,255,255,255),
        ),
        layout=Layout(
            # You can easily move pieces around here now:
            # lens_cx=0.65,
            # handle_attach_angle_deg=40,
        ),
    )
