#!/usr/bin/env python3
"""Run the YOLOX-nano poop ONNX model in Python and print the top
detections, so the Kotlin pipeline can be cross-checked for parity.

Usage:
    python scripts/python_reference_compare.py \\
        --image tpl/YOLOX/assets/dog.jpg \\
        --model tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx \\
        --top 5 --threshold 0.25

This intentionally re-implements the same letterbox + YOLOX postprocess
that lives in `composeApp/src/commonMain/.../Preprocessing.kt` and
`Yolox.kt`. If the two stop agreeing on the top-N detections you have
either (a) drift in the Kotlin postprocess, (b) precision drift in
the model export, or (c) drift in this script. Update whichever side
is wrong; do not "fix" the other side to match.

Requires `onnxruntime` and `pillow`. The shitspotter-app-toolchain VM
has a venv at /tmp/onnx_venv that includes both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def letterbox(img: Image.Image, dst: int, pad_value: int = 114):
    src_w, src_h = img.size
    scale = min(dst / src_w, dst / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    pad_x = (dst - new_w) // 2
    pad_y = (dst - new_h) // 2
    resized = img.resize((new_w, new_h))
    canvas = Image.new("RGB", (dst, dst), (pad_value,) * 3)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def to_nchw(canvas: Image.Image) -> np.ndarray:
    arr = np.array(canvas, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)[None, :, :, :]
    return arr


def yolox_postprocess(preds: np.ndarray, num_classes: int = 1, threshold: float = 0.25):
    """preds shape [num_anchors, 4 + 1 + num_classes]."""
    box = preds[:, 0:4]
    obj = preds[:, 4]
    cls = preds[:, 5 : 5 + num_classes]
    cls_id = cls.argmax(axis=1)
    cls_score = cls.max(axis=1)
    score = obj * cls_score
    mask = score >= threshold
    return box[mask], score[mask], cls_id[mask]


def map_back(box, scale, pad_x, pad_y):
    cx, cy, w, h = box
    bx = (cx - w / 2 - pad_x) / scale
    by = (cy - h / 2 - pad_y) / scale
    bw = w / scale
    bh = h / scale
    return bx, by, bw, bh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--input-size", type=int, default=416)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    img = Image.open(args.image).convert("RGB")
    canvas, scale, pad_x, pad_y = letterbox(img, args.input_size)
    inp = to_nchw(canvas)

    s = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = s.get_inputs()[0].name
    out = s.run(None, {in_name: inp})[0]
    print(f"# python reference — {args.model}")
    print(f"# image    : {args.image} ({img.size[0]}x{img.size[1]})")
    print(f"# input    : {args.input_size}x{args.input_size}")
    print(f"# threshold: {args.threshold}")
    print(f"# raw output shape : {tuple(out.shape)}")

    boxes, scores, cls_ids = yolox_postprocess(out[0], args.num_classes, args.threshold)
    print(f"# above-threshold  : {len(scores)}")
    order = np.argsort(-scores)
    rows = []
    for rank in range(min(args.top, len(scores))):
        i = order[rank]
        bx, by, bw, bh = map_back(boxes[i], scale, pad_x, pad_y)
        rows.append(
            dict(
                rank=rank,
                score=float(scores[i]),
                class_id=int(cls_ids[i]),
                box=[float(bx), float(by), float(bw), float(bh)],
            )
        )
        print(
            f"  det[{rank}] score={float(scores[i]):.4f} "
            f"class={int(cls_ids[i])} "
            f"box=(x={bx:.1f}, y={by:.1f}, w={bw:.1f}, h={bh:.1f})"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"rows": rows}, indent=2))
        print(f"# wrote {args.json_out}")


if __name__ == "__main__":
    main()
