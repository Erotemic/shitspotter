# 002 — Benchmark template (per GOAL.md "Performance discipline")

This doc is the schema for a single benchmark report. Future agents and
the user should append concrete reports as new files in this folder
(e.g. `docs/benchmarks/2026-05-10_pixel5_yolox_nano.md`) — never edit
this template in place.

The shared core defines this same schema in [`BackendComparison.kt`](../composeApp/src/commonMain/kotlin/io/kitware/shitspotter/core/BackendComparison.kt)
so the desktop CLI can emit a JSON report that drops directly into a
markdown table.

## Required fields per row

```text
device   | os version | build (debug/release) | model id     | runtime          | delegate | input    | capture | preprocess | inference | postprocess | overlay | fps   | dets
```

- `device` — `Build.MANUFACTURER + " " + Build.MODEL` on Android, or
  `Linux desktop (...)` on the desktop harness.
- `os version` — `"Android <release> (sdk=<level>)"` or kernel version.
- `build` — `debug` (assembleDebug, no proguard) or `release` once we
  add a release build type.
- `model id` — value of `ModelSpec.modelId` (e.g.
  `yolox-nano-poop-cropped-v1`).
- `runtime` — backendName field (`onnxruntime-android-1.19`,
  `onnxruntime-jvm-1.19`, etc.).
- `delegate` — value reported by `OnnxRuntimeAndroidBackend.delegate`
  (`NNAPI`, `CPU`); never make this up.
- `input` — `<inputWidth>x<inputHeight>` from the active `ModelSpec`.
- `capture` — milliseconds to copy the camera frame into our
  `ImageProxyFrame` wrapper.
- `preprocess` — milliseconds to letterbox + RGB-to-tensor.
- `inference` — milliseconds spent inside `session.run`.
- `postprocess` — milliseconds for decode + NMS + letterbox-undo.
- `overlay` — milliseconds for the overlay composable to redraw.
- `fps` — sliding-window FPS reported by `FpsCounter`.
- `dets` — number of detections kept after NMS at the configured
  threshold.

Always also note **whether preview remained smooth**. A 30 FPS
inference number is meaningless if the camera preview was stalling.

## Generating a JSON report from the desktop harness

```bash
cd tpl/shitspotter-phone-app
./gradlew :composeApp:run --args="compare \
   --image=/path/to/test.jpg \
   --model=/path/to/yolox_nano_poop_cropped_only_best.onnx \
   --runs=10 --warmup=2 \
   --out=docs/benchmarks/<DATE>_desktop_<image-name>.json"
```

The CLI prints a markdown-style table to stdout and writes a JSON
report at `--out`. Drop both into the report alongside the human notes.

## Sample report (Linux desktop, dog.jpg, YOLOX-nano poop)

This is a real run captured during the Milestone 1/2/3 scaffolding. It
is intentionally a non-poop image so the score-threshold drift is
visible. It is **not** a phone benchmark — Pixel 5 numbers go in their
own report once a user runs them.

```text
# image:  tpl/YOLOX/assets/dog.jpg (768x576)
# device: Linux desktop (Linux amd64)
# os:     6.8.0-110-generic
# runs:   5 (warmup=2)

backend                    | delegate |  pre(ms) |  inf(ms) | post(ms) |  dets |     top
----------------------------------------------------------------------------------------
stub-1.0                   | —        |     1.05 |     0.00 |     0.00 |     1 |   0.880
onnxruntime-jvm-1.19       | CPU      |     4.54 |     9.91 |     2.82 |     3 |   0.757
```

Observations from this run, recorded so future reports follow the same
form:

- The dog image lights up YOLOX-nano poop with a 0.757 false positive at
  the default 0.25 threshold. This is **expected** — the model is
  cropped-only and easily confused on out-of-distribution textures.
  Pixel 5 reports should call this out so it doesn't get reported as
  a regression later.
- 9.91 ms inference at 416x416 on a desktop CPU is roughly the right
  order of magnitude for a YOLOX-nano on JVM. NNAPI on Pixel 5 is the
  unknown — that's exactly what the on-device benchmark needs to measure.
- The stub backend reports 0 inference cost, as it should.

## Pixel 5 report stub

Copy this block into a new file the first time you run on a Pixel 5:

```markdown
# <DATE> — Pixel 5 benchmark — YOLOX-nano poop

Device: Pixel 5 (Snapdragon 765G, Adreno 620, 8 GB)
OS:     Android <release> (sdk=<level>)
APK:    composeApp-debug.apk (debug, unsigned)
Model:  yolox_nano_poop_cropped_only_best.onnx (sha256: …)
Test:   pointing the camera at <subject> for ~30 s

Average across the last ~30 s of the HUD readings:

| backend              | delegate | input   | capture | preprocess | inference | postprocess | overlay | fps  | dets |
|----------------------|----------|---------|---------|------------|-----------|-------------|---------|------|------|
| onnxruntime-android  | NNAPI    | 416x416 |         |            |           |             |         |      |      |
| onnxruntime-android  | CPU      | 416x416 |         |            |           |             |         |      |      |

Preview smoothness: ___
Notes:
- delegate fallback fired? (yes/no)
- camera preview at 720p? 1080p?
- thermal: hot to the touch after N minutes? (yes/no)
- battery: drop per minute estimate
- any logcat exceptions tagged ShitSpotter.*

Compared against GOAL.md targets:
- ≥ 1 FPS minimum     ___
- ≥ 10 FPS desired    ___
- 15-30 FPS excellent ___
```

## Where reports live

- `tpl/shitspotter-phone-app/docs/benchmarks/<DATE>_<device>_<model>.md`
  is the human-readable summary.
- The JSON sibling (same basename, `.json`) is committed alongside; the
  schema matches `BackendComparisonReport`.
- The JPEG/snapshot used to generate the report does **not** get
  committed (binary data rule).
