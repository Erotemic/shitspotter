#!/usr/bin/env python3
"""
Prerequisites (Linux / venv) — render *color* emoji with Pango+Cairo (gi/PyGObject)

This script uses Pango+Cairo via GObject Introspection (`gi`) to render emoji in
full color (e.g., Noto Color Emoji). Pillow alone often renders NotoColorEmoji
monochrome on Linux, so these system deps are required.

System packages (Debian/Ubuntu):
    sudo apt-get update
    sudo apt-get install -y \
        pkg-config \
        gobject-introspection \
        libgirepository-2.0-dev \
        gir1.2-pango-1.0 \
        libcairo2-dev

Notes:
  - `libgirepository-2.0-dev` is what provides the `girepository-2.0` dependency
    that PyGObject needs when building inside a venv.
  - You may also want the emoji font:
        sudo apt-get install -y fonts-noto-color-emoji

Python packages (inside your venv):
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install qrcode[pil] pillow pycairo PyGObject

Quick sanity check (inside venv):
    python -c "import gi; gi.require_version('Pango','1.0'); from gi.repository import Pango; print('OK', Pango)"

If you prefer avoiding PyGObject builds:
  - You can install `python3-gi` + `python3-gi-cairo` from apt and create your venv
    with system site packages:
        sudo apt-get install -y python3-gi python3-gi-cairo
        python3 -m venv --system-site-packages .venv
"""
import math
import qrcode
from PIL import Image, ImageDraw

import gi
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo
import cairo

# Hardcoded settings
URL = "https://github.com/Erotemic/scatspotter"
OUT_PATH = "out.png"
EMOJI = "💩"

def render_color_emoji_pango(emoji: str, px: int, font_family: str = "Noto Color Emoji") -> Image.Image:
    """
    Render a color emoji into a PIL RGBA image using Pango+Cairo.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, px, px)
    ctx = cairo.Context(surface)

    # Transparent background
    ctx.set_source_rgba(1, 1, 1, 0)
    ctx.paint()

    layout = PangoCairo.create_layout(ctx)
    layout.set_text(emoji, -1)

    # Font size is in Pango units; 1 pt = Pango.SCALE units. We'll pick a big size
    # and let Pango lay it out, then center it.
    # The numeric size here is "points", but practically it maps well for our px canvas.
    desc = Pango.FontDescription(f"{font_family} {int(px * 0.75)}")
    layout.set_font_description(desc)

    ink, logical = layout.get_pixel_extents()
    x = (px - logical.width) / 2 - logical.x
    y = (px - logical.height) / 2 - logical.y
    ctx.move_to(x, y)
    PangoCairo.show_layout(ctx, layout)

    # Convert Cairo surface -> PIL Image
    buf = surface.get_data()  # BGRA premultiplied
    img = Image.frombuffer("RGBA", (px, px), buf, "raw", "BGRA", 0, 1)
    return img.copy()  # detach from Cairo buffer

def main():
    # Build QR code
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=12,
        border=4,
    )
    qr.add_data(URL)
    qr.make(fit=True)

    base = qr.make_image(fill_color="black", back_color="white").convert("RGBA")
    w, h = base.size

    # Overlay size (keep modest)
    overlay_size = int(min(w, h) * 0.33)

    # Create overlay with white rounded background
    overlay = Image.new("RGBA", (overlay_size, overlay_size), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    margin = int(overlay_size * 0.06)
    d.rounded_rectangle(
        [margin, margin, overlay_size - margin, overlay_size - margin],
        radius=int(overlay_size * 0.22),
        fill=(255, 255, 255, 255),
    )

    emoji_img = render_color_emoji_pango(EMOJI, int(overlay_size * 0.82), font_family="Noto Color Emoji")
    px = (overlay_size - emoji_img.size[0]) // 2
    py = (overlay_size - emoji_img.size[1]) // 2
    overlay.alpha_composite(emoji_img, dest=(px, py))

    # Composite overlay into center of QR
    pos = ((w - overlay_size) // 2, (h - overlay_size) // 2)
    base.alpha_composite(overlay, dest=pos)

    base.convert("RGB").save(OUT_PATH, "PNG")

    import kwplot
    kwplot.cropwhite_ondisk(OUT_PATH)
    print(f"Saved {OUT_PATH} linking to {URL}")

if __name__ == "__main__":
    main()
