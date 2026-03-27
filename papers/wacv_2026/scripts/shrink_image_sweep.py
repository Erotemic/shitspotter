#!/usr/bin/env python3
"""
Generate candidate "shrunk" variants of an image to evaluate file size tradeoffs.

- For JPEG inputs (or if you choose to write JPEG), it sweeps JPEG quality.
- Optionally sweeps scale factors and/or max width/height.
- Writes all outputs to an output directory and prints a summary.

Examples:
  python image_shrink_sweep.py path/to/image.jpg

  # Sweep qualities and downscale factors
  python image_shrink_sweep.py path/to/image.jpg --qualities 95,90,85,80,75 --scales 1.0,0.75,0.5,0.33

  # Cap width while sweeping quality
  python image_shrink_sweep.py path/to/image.jpg --max-width 6000 --qualities 92,88,84,80

  # Force output format
  python image_shrink_sweep.py path/to/image.png --out-format jpg --qualities 92,85,78 --scales 1,0.5


  python image_shrink_sweep.py /home/joncrall/code/shitspotter/papers/wacv_2026/_arxiv_submission/figures/agg_viz_results2/test_imgs121_6cb3b6ff.kwcoco/results_input_images.jpg --qualities 95,90,85,80,75 --scales 1.0,0.75,0.5,0.33
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    from PIL import Image
except Exception as ex:
    raise SystemExit(
        "This script requires Pillow.\n"
        "Install it with: pip install pillow\n"
        f"Import error: {ex}"
    )


@dataclass(frozen=True)
class Variant:
    scale: float
    max_width: Optional[int]
    max_height: Optional[int]
    quality: Optional[int]
    out_format: str  # "jpg" or "png"
    progressive: bool
    optimize: bool


def human_bytes(n: int) -> str:
    # Simple base-2 formatting
    units = ["B", "KiB", "MiB", "GiB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.1f} {u}" if u != "B" else f"{int(v)} {u}"
        v /= 1024
    return f"{n} B"


def compute_size(w: int, h: int, scale: float, max_w: Optional[int], max_h: Optional[int]) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("scale must be > 0")

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Apply max caps after scaling
    cap_scale = 1.0
    if max_w is not None and new_w > max_w:
        cap_scale = min(cap_scale, max_w / float(new_w))
    if max_h is not None and new_h > max_h:
        cap_scale = min(cap_scale, max_h / float(new_h))

    if cap_scale < 1.0:
        new_w = max(1, int(round(new_w * cap_scale)))
        new_h = max(1, int(round(new_h * cap_scale)))

    return new_w, new_h


def save_variant(
    im: Image.Image,
    src_path: Path,
    out_dir: Path,
    var: Variant,
    base_w: int,
    base_h: int,
) -> tuple[Path, int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    target_w, target_h = compute_size(base_w, base_h, var.scale, var.max_width, var.max_height)
    out_im = im

    if (target_w, target_h) != (base_w, base_h):
        out_im = im.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    stem = src_path.stem
    fmt = var.out_format.lower()
    qtag = f"q{var.quality}" if var.quality is not None else "qNA"
    stag = f"s{var.scale:g}"
    wtag = f"mw{var.max_width}" if var.max_width else "mwNA"
    htag = f"mh{var.max_height}" if var.max_height else "mhNA"
    fname = f"{stem}__{fmt}__{stag}__{wtag}__{htag}__{qtag}.{fmt}"
    out_path = out_dir / fname

    save_kwargs = {}
    if fmt in ("jpg", "jpeg"):
        # Ensure JPEG-compatible mode
        if out_im.mode not in ("RGB",):
            out_im = out_im.convert("RGB")
        save_kwargs["format"] = "JPEG"
        if var.quality is not None:
            save_kwargs["quality"] = int(var.quality)
        save_kwargs["optimize"] = bool(var.optimize)
        save_kwargs["progressive"] = bool(var.progressive)
    elif fmt == "png":
        save_kwargs["format"] = "PNG"
        # Pillow optimize is lossless; can be slow on big images
        save_kwargs["optimize"] = bool(var.optimize)
    else:
        raise ValueError(f"Unsupported out_format={var.out_format!r} (use jpg or png)")

    out_im.save(out_path, **save_kwargs)
    size_bytes = out_path.stat().st_size
    return out_path, target_w, target_h, size_bytes


def parse_floats_csv(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_ints_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate candidate shrunk variants of an image.")
    ap.add_argument("image", type=Path, help="Input image path (jpg/png).")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <image>__shrink_sweep/).")
    ap.add_argument("--out-format", type=str, default=None, help="Force output format: jpg or png (default: same as input).")

    ap.add_argument("--qualities", type=str, default="95,90,85,80,75,70",
                    help="Comma-separated JPEG qualities to try (ignored for png output unless you force jpg).")
    ap.add_argument("--scales", type=str, default="1.0,0.75,0.5,0.33",
                    help="Comma-separated scale factors to try, e.g. 1.0,0.75,0.5")

    ap.add_argument("--max-width", type=int, default=None, help="Optional max width cap (applies in addition to scale).")
    ap.add_argument("--max-height", type=int, default=None, help="Optional max height cap (applies in addition to scale).")

    ap.add_argument("--no-progressive", action="store_true", help="Disable progressive JPEG.")
    ap.add_argument("--no-optimize", action="store_true", help="Disable encoder optimize flag.")

    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of variants produced.")
    args = ap.parse_args(argv)

    src = args.image
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    in_ext = src.suffix.lower().lstrip(".")
    default_fmt = "jpg" if in_ext in ("jpg", "jpeg") else ("png" if in_ext == "png" else "jpg")
    out_fmt = (args.out_format.lower() if args.out_format else default_fmt)
    if out_fmt not in ("jpg", "jpeg", "png"):
        raise SystemExit("out-format must be one of: jpg, png")

    out_dir = args.out_dir if args.out_dir else (src.parent / f"{src.stem}__shrink_sweep")

    qualities = parse_ints_csv(args.qualities)
    scales = parse_floats_csv(args.scales)

    # Build variants
    variants: list[Variant] = []
    if out_fmt in ("jpg", "jpeg"):
        for s in scales:
            for q in qualities:
                variants.append(
                    Variant(
                        scale=float(s),
                        max_width=args.max_width,
                        max_height=args.max_height,
                        quality=int(q),
                        out_format="jpg",
                        progressive=(not args.no_progressive),
                        optimize=(not args.no_optimize),
                    )
                )
    else:
        # PNG output: qualities don't apply, but scale does
        for s in scales:
            variants.append(
                Variant(
                    scale=float(s),
                    max_width=args.max_width,
                    max_height=args.max_height,
                    quality=None,
                    out_format="png",
                    progressive=False,
                    optimize=(not args.no_optimize),
                )
            )

    if args.limit is not None:
        variants = variants[: max(0, int(args.limit))]

    # Load input
    with Image.open(src) as im:
        base_w, base_h = im.size

        print(f"Input: {src}")
        print(f"  format={im.format} mode={im.mode} size={base_w}x{base_h} bytes={src.stat().st_size} ({human_bytes(src.stat().st_size)})")
        print(f"Output dir: {out_dir}")
        print("")

        rows = []
        for var in variants:
            out_path, w, h, b = save_variant(im, src, out_dir, var, base_w, base_h)
            rows.append((var.out_format, var.scale, w, h, var.quality, b, out_path.name))

        # Print table sorted by size
        rows.sort(key=lambda r: r[5])

        header = f"{'fmt':>4}  {'scale':>6}  {'WxH':>12}  {'qual':>4}  {'bytes':>12}  {'human':>10}  name"
        print(header)
        print("-" * len(header))
        for fmt, scale, w, h, qual, b, name in rows:
            qtxt = f"{qual:>4}" if qual is not None else "   -"
            print(f"{fmt:>4}  {scale:>6.2f}  {w:>5}x{h:<5}  {qtxt}  {b:>12}  {human_bytes(b):>10}  {name}")

        print("")
        print("Tip: pick a candidate and visually compare it with the original (e.g. using an image viewer).")


if __name__ == "__main__":
    main()
