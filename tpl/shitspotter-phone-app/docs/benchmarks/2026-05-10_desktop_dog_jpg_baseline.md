# 2026-05-10 — Linux desktop baseline (dog.jpg, JVM CPU)

**This is the parity / pipeline-shape baseline, not a phone benchmark.**
The model is intentionally pointed at an out-of-distribution image so
any regression in pre/post-processing or NMS is loud. The latency
numbers are from JVM CPU, which is also intentional — Pixel 5 NNAPI
numbers go in a sibling file once the user runs the APK.

## Inputs

```text
host:    Linux desktop (Linux amd64), 6.8.0-110-generic
runtime: ONNX Runtime JVM 1.19.2, CPU EP
image:   tpl/YOLOX/assets/dog.jpg (768x576)
runs:    10 measured, 2 warmup
threshold: 0.25 (model default)
```

## Multi-model comparison (3-model run)

```text
model                            | backend              | delegate |  pre(ms) |  inf(ms) | post(ms) | dets |  top
----------------------------------------------------------------------------------------------------------------
yolox-nano-poop-cropped-v1       | onnxruntime-jvm-1.19 | CPU      |     4.54 |     9.91 |     2.82 |    3 |.757
shitspotter-custom-v5-epoch115   | onnxruntime-jvm-1.19 | CPU      |     8.35 |    49.92 |     1.39 |    0 |.000
shitspotter-custom-v2-epoch126   | onnxruntime-jvm-1.19 | CPU      |    10.68 |    49.25 |     1.85 |    0 |.000
```

Source JSON: `dev/journals/2026-05-10_phone_app_compare_3_models_dog.json`.

## Stub baseline

```text
model                | backend  | delegate | pre(ms) | inf(ms) | post(ms) | dets | top
--------------------------------------------------------------------------------------
stub-fake-detector   | stub-1.0 | —        |    1.05 |    0.00 |    0.00 |    1 |.880
```

## Python parity (NumPy + onnxruntime CPU)

Same model + same image + threshold=0.25:

```text
raw above-threshold: 8 (no NMS)
top score:           0.7463
top box:             (-1.6, -97.5, 1.9, 2.8)  # degenerate; out-of-distribution
```

Vs Kotlin pipeline post-NMS: 3 detections, top score 0.757. Within
the precision drift expected from Pillow vs java.awt.image
letterboxing. See `../004_kotlin_python_parity.md` for the full
discussion.

## What this baseline does NOT measure

- NNAPI / GPU / NPU latency on real hardware. **Run on Pixel 5 next.**
- p50/p90/p99 latency under sustained load. The 10-run mean is fine
  for catching gross regressions but will not tell us whether a
  particular delegate is throttling.
- Battery/thermal behaviour. Cannot measure without a phone.
- True FPS. The CompareCli does single-frame inference back-to-back;
  the camera path adds capture + overlay + UI cost on top.

## Fields the next benchmark file should add

When the user runs on Pixel 5, copy this file's structure but include:

- delegate that actually loaded (NNAPI vs CPU EP) per the HUD
- preview-smoothness note (subjective)
- thermal/battery observation
- whether any logcat exceptions appeared tagged `ShitSpotter.*`
- HUD `dropped` counter at end of run (should be 0 if you're not
  pausing, indicating the analyzer kept up with the camera)
