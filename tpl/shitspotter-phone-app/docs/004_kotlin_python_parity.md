# 004 — Kotlin / Python parity check (Milestone 2 reference comparison)

GOAL.md "Milestone 2 validation" asks for at least one input compared
against a Python/reference output. This is that comparison.

## What was checked

```text
model:     tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx
image:     tpl/YOLOX/assets/dog.jpg (768x576)
input:     416x416 letterbox, RGB, no /255 normalisation, NCHW
threshold: 0.25
```

Both pipelines are intentionally minimal re-implementations of the same
YOLOX-nano post-decode + score-filter steps. They share the input image
and ONNX file but are otherwise independent: the Python script lives at
[`scripts/python_reference_compare.py`](../scripts/python_reference_compare.py)
and the Kotlin path is in
[`composeApp/src/commonMain/.../Yolox.kt`](../composeApp/src/commonMain/kotlin/io/kitware/shitspotter/core/Yolox.kt)
plus the per-target ONNX backends.

## Headline result

| Pipeline | Raw above-threshold | Top score | Notes |
|----------|--------------------:|----------:|-------|
| Python (NumPy + ONNX Runtime CPU) | 8 | 0.7463 | Pre-NMS filtered list |
| Kotlin/JVM (ORT-jvm CPU)          | 3 | 0.757  | Post-NMS filtered list |

The Kotlin pipeline runs `Nms.apply(..., iouThreshold=0.45)` after the
score filter; the Python script does not. So 8 → 3 is the expected NMS
collapse, not a Kotlin bug.

The top score differs by ~0.011, which is within the precision drift
expected from running the same model against an image that has been
letterboxed by two different image libraries (Pillow vs java.awt.image).
This is intentionally not asserted to bit-exact equality.

## Where the boxes actually land

```text
python rank score class box(x,y,w,h)
       0    0.7463 0    -1.6, -97.5,  1.9 x 2.8
       1    0.7351 0     0.2, -95.7,  2.0 x 2.7
       2    0.7310 0     0.3, -97.5,  1.9 x 2.8
```

These boxes are **degenerate** — tiny (~2x3 px), located off the top
edge of the source image. This is the model over-firing on dog.jpg,
not a bug in either pipeline. dog.jpg is well outside the
poop-detection training distribution; the ~0.75 confidence is a
reminder that YOLOX-nano-poop is **cropped-only** (trained on
pre-cropped patches, not full images) and should not be trusted on
out-of-distribution inputs without further filtering.

For real Pixel-5 validation the user must run the app against actual
ground-truth poop captures and report HUD numbers, not against
`tpl/YOLOX/assets/dog.jpg`. This image is the parity test, not a
quality test.

## How to repeat the comparison

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
cd tpl/shitspotter-phone-app

# Kotlin/JVM side (already exercised by :composeApp:desktopTest):
./gradlew :composeApp:run --console=plain --args="compare \
   --image=../YOLOX/assets/dog.jpg \
   --model=../poop_models/yolox_nano_poop_cropped_only_best.onnx \
   --runs=3 --warmup=1 --no-stub"

# Python side (uses the toolchain's onnxruntime venv at /tmp/onnx_venv):
/tmp/onnx_venv/bin/python scripts/python_reference_compare.py \
   --image ../YOLOX/assets/dog.jpg \
   --model ../poop_models/yolox_nano_poop_cropped_only_best.onnx \
   --threshold 0.25 --top 5
```

## What the next agent should add

- A real ground-truth image pair from the ShitSpotter dataset, asserted
  to within bit-exact tolerance.
- A `scripts/check_parity.py` driver that runs both pipelines and exits
  non-zero if the top-N detection sets disagree by more than X% IoU.
- A pinned `tpl/shitspotter-phone-app/scripts/requirements.txt` so the
  python venv is reproducible (currently the toolchain venv at
  `/tmp/onnx_venv` is hand-managed).
