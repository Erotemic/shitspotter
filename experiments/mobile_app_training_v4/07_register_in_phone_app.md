# Step 7 — Register v4 DEIMv2 candidates in the Pixel 5 phone app

This document is **prescriptive** for the next agent (or you). It does
NOT need to run on the VM — it's the changeset to apply on the host
machine where the phone app is built and the Pixel 5 is sideloaded.

Each v4 sweep cell becomes its **own** `ModelSpec` in the registry.
There is no "one DEIMv2 model" — `deimv2_n_320`, `deimv2_n_416`,
`deimv2_pico_416`, etc. are distinct entries because the winner of the
Pareto sweep is resolution-dependent (e.g. `deimv2_pico_512` may beat
`deimv2_n_320` on quality while still passing the FPS gate).

The eligibility manifest tells you which entries to register and which
to skip — `phone_model_id` and `pixel5_eligible` are the keys.

## Why this is its own step

The current phone app (`tpl/shitspotter-phone-app/`) ships postprocess
parsers for `YOLOX`, `YOLO_V9`, `YOLO_V9_DFL`, and a generic box/score
parser. DEIMv2 is structurally different:

* Two inputs: `images: NxCxHxW float32` and `orig_target_sizes: Nx2 int64`.
* Three outputs: `labels`, `boxes`, `scores`. Boxes are already in **pixel
  coordinates** with respect to `orig_target_sizes`, not in grid units.
* No anchors, no DFL distribution, no objectness — `scores` is the
  per-query confidence after the built-in postprocessor.
* **Fixed input shape per export.** Each `(variant, export_h, export_w)`
  produces its own ONNX. Dynamic-shape ONNX is a future step; for now
  every registered entry has explicit `inputWidth`/`inputHeight`.

That means the existing YOLO parsers will mis-decode DEIMv2 outputs.
Add a new `PostprocessType.DEIMV2` so the model registry can dispatch
correctly.

## Changes

### 1. Add the postprocess type

Edit
[`tpl/shitspotter-phone-app/composeApp/src/commonMain/kotlin/io/kitware/shitspotter/core/ModelSpec.kt`](../../tpl/shitspotter-phone-app/composeApp/src/commonMain/kotlin/io/kitware/shitspotter/core/ModelSpec.kt):

```kotlin
@Serializable
enum class PostprocessType {
    YOLOX,
    YOLO_V9,
    YOLO_V9_DFL,
    DEIMV2,                       // <-- add
    GENERIC_BOX_SCORE_CLASS,
    NONE,
    STUB,
}
```

Add a schema record alongside `Yolov9Schema`:

```kotlin
@Serializable
data class Deimv2Schema(
    val imagesInput: String = "images",
    val origSizeInput: String = "orig_target_sizes",
    val labelsOutput: String = "labels",
    val boxesOutput: String = "boxes",
    val scoresOutput: String = "scores",
    val passOrigSize: Boolean = true,
)
```

…and a nullable field on `ModelSpec`:

```kotlin
val deimv2Schema: Deimv2Schema? = null,
```

### 2. Add one registry entry per eligible candidate

The eligibility manifest emits one `phone_model_id` per row, e.g.
`shitspotter-deimv2_n-h320w320-multiscale_256_416`. Add one
`ModelSpec` per row whose `pixel5_eligible` is `yes` (or `TODO` if
you're sideloading to *measure* it). Most of the fields come straight
from the `*.modelspec.json` sidecar that `03_export_onnx.sh` wrote
next to the ONNX file.

Example for the `deimv2_n` 320x320 cell:

```kotlin
val DEIMV2_N_320 = ModelSpec(
    modelId = "shitspotter-deimv2_n-h320w320-multiscale_256_416",
    displayName = "DEIMv2-N 320x320 (multiscale 256-416)",
    modelFile = "deimv2_n_h320_w320.onnx",
    format = ModelFormat.ONNX,
    inputWidth = 320,
    inputHeight = 320,
    inputLayout = InputLayout.NCHW,
    colorOrder = ColorOrder.RGB,
    normalization = Normalization(scale = 1f / 255f),
    resizePolicy = ResizePolicy.LETTERBOX,
    postprocessType = PostprocessType.DEIMV2,
    classNames = listOf("poop"),
    scoreThreshold = 0.30f,
    iouThreshold = 0.45f,
    deimv2Schema = Deimv2Schema(),
    notes = "v4 sweep cell. Two inputs (images, orig_target_sizes); " +
            "three outputs in pixel coords. Trained at multi-scale " +
            "(256..416) so the same checkpoint is robust around 320 input.",
)
```

Repeat for each eligible cell — `_416`, `_512`, etc. — and for the
`deimv2_pico_*` family. Consider grouping them in the chip row so the
user sees them as a family of choices, not a single entry.

Add each to `ModelRegistry.all`. Keep STUB at index 0 so the app's
default remains the safe one until the user picks a real model.

### 3. Implement the DEIMv2 backend path

`composeApp/src/androidMain/kotlin/io/kitware/shitspotter/onnx/OnnxDetectorBackend.kt`
(or the equivalent JVM/desktop file) currently constructs a single-input
session and runs YOLO postprocess. Add a branch:

```kotlin
PostprocessType.DEIMV2 -> {
    val schema = spec.deimv2Schema!!
    val origSize = LongArray(2) { i -> if (i == 0) frameW.toLong() else frameH.toLong() }
    val inputs = mapOf(
        schema.imagesInput to OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, spec.inputHeight.toLong(), spec.inputWidth.toLong())),
        schema.origSizeInput to OnnxTensor.createTensor(env, LongBuffer.wrap(origSize), longArrayOf(1, 2)),
    )
    val results = session.run(inputs)
    val labels = results[schema.labelsOutput] as OnnxTensor
    val boxes  = results[schema.boxesOutput] as OnnxTensor
    val scores = results[schema.scoresOutput] as OnnxTensor
    decodeDeimv2(labels, boxes, scores, spec.scoreThreshold, spec.iouThreshold, spec.classNames)
}
```

`decodeDeimv2(...)` should:

1. Iterate over the (typically 300) per-query detections.
2. Filter by `score >= scoreThreshold`.
3. Convert `boxes[i] = [x0, y0, x1, y1]` from `(W, H)` pixel space into
   the overlay's coordinate frame the same way the YOLOX path does
   (i.e. apply the inverse letterbox + the camera rotation).
4. Optionally apply class-agnostic NMS at `iouThreshold`. The DEIMv2
   postprocessor already performs top-K selection, so a strict NMS is
   often unnecessary, but the existing app uses NMS uniformly.

### 4. Validate parity from the host

Run `python 05_desktop_onnx_parity.py` *on the host*, against `dog.jpg`
and a positive shitspotter test image, before pushing the new model to
the phone. The parity test catches the same "shape / output-name" class
of bug that bit the YOLOX side at deploy time
(`dev/journals/lessons_learned.md` §1).

### 5. Add the model files to the sideload bundle

The phone app reads models from `tpl/poop_models/` (a separate
submodule). The `*.onnx` artifacts in that folder are gitignored — copy
each eligible v4 export there manually, or via the existing
`scripts/install_to_phone.sh` flow. The eligibility manifest's
`onnx_path` column tells you which files to copy.

```bash
# All eligible exports in one go (filter the manifest however you like)
"$PYTHON_BIN" - <<'PY'
import csv, shutil, os
from pathlib import Path
manifest = Path(os.environ['V4_ROOT']) / 'manifest.tsv'
dst = Path(os.environ['SHITSPOTTER_DPATH']) / 'tpl/poop_models'
with manifest.open() as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if row.get('pixel5_eligible') == 'no':
            continue
        onnx = row.get('onnx_path')
        if onnx and Path(onnx).exists():
            print('cp', onnx)
            shutil.copy2(onnx, dst)
PY
```

### 6. Run on-device benchmark

Use `05_bench_on_pixel5.sh` — it compiles a tiny C benchmark (`ort_bench.c`)
against the ORT 1.19 ARM64 shared library already in the Gradle cache, pushes
it to the device, and writes `$V4_ROOT/pixel5_bench.tsv`:

```bash
# on the workstation with the Pixel 5 plugged in
source /data/tmp/shitspotter-app-toolchain/env.sh
V4_ROOT=/data/joncrall/shitspotter_v4 \
    bash experiments/mobile_app_training_v4/05_bench_on_pixel5.sh
```

The script benchmarks all four Pareto-front cells (pico@320, pico@416,
n@512, n@640) with NNAPI EP (falling back to CPU) and emits a TSV that
`eligibility_manifest.py --pixel5_index` consumes directly.

Once sideloaded with DEIMv2 support (steps 1–5 above), also exercise:

```text
1. Live FPS for FAST_FULL_FRAME at the new model.
2. Latency p50/p99 logged from the app telemetry.
3. Failure-case capture on at least one true positive and the dog.jpg
   negative — make sure the new model is not over-firing in OOD scenes.
```

Acceptance bar (from the dual advisor briefs in the v4 README):

```text
Pixel 5 live camera, FAST_FULL_FRAME mode:
  >= 10 FPS sustained
  preview remains smooth
  no unbounded frame queue
  model latency + preprocessing + postprocess under 100 ms/frame
  failure-case capture still works
```

If a candidate misses the bar, drop the next-cheapest cell from the
sweep into the registry the same way. The eligibility manifest is the
single source of truth for which entries belong on the phone.

After running the on-device benchmark, write a TSV like:

```text
candidate_id	latency_ms	fps
shitspotter-deimv2_n-h320w320-multiscale_256_416	58.2	17.2
shitspotter-deimv2_n-h416w416-multiscale_320_512	102.5	9.8
```

…and feed it back into the manifest:

```bash
"$PYTHON_BIN" experiments/mobile_app_training_v4/eligibility_manifest.py \
    --auto \
    --pixel5_index "$V4_ROOT/pixel5_bench.tsv" \
    --max_desktop_ms 80 \
    --min_pixel5_fps 10 \
    --out "$V4_ROOT/manifest.tsv"
```

The script then prints the eligible winner — the highest-AP candidate
that passes both gates.

---

## Where the docs and lessons land

Each on-phone failure or surprise should:

* Add a new entry in
  `tpl/shitspotter-phone-app/dev/benchmark-candidates/app-deployment-questions.md`
* Append to `dev/journals/lessons_learned.md` if it's a new failure
  category (don't over-grow the file with restatements of existing
  invariants — link to the existing entry instead).
