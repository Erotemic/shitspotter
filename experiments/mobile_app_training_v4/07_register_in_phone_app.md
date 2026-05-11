# Step 7 — Register a v4 DEIMv2 model in the Pixel 5 phone app

This document is **prescriptive** for the next agent (or you). It does
NOT need to run on the VM — it's the changeset to apply on the host
machine where the phone app is built and the Pixel 5 is sideloaded.

## Why this is its own step

The current phone app (`tpl/shitspotter-phone-app/`) ships postprocess
parsers for `YOLOX`, `YOLO_V9`, `YOLO_V9_DFL`, and a generic box/score
parser. DEIMv2 is structurally different:

* Two inputs: `images: NxCxHxW float32` and `orig_target_sizes: Nx2 int64`.
* Three outputs: `labels`, `boxes`, `scores`. Boxes are already in **pixel
  coordinates** with respect to `orig_target_sizes`, not in grid units.
* No anchors, no DFL distribution, no objectness — `scores` is the
  per-query confidence after the built-in postprocessor.

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

### 2. Add a registry entry

Add a static `ModelSpec` companion entry whose values come straight from
the `*.modelspec.json` sidecar that `03_export_onnx.sh` wrote next to
the ONNX file. Example for `deimv2_n` at 320x320:

```kotlin
val DEIMV2_N_TILE_G2_320 = ModelSpec(
    modelId = "shitspotter-deimv2_n-tile_g2-h320w320",
    displayName = "DEIMv2-N tile_g2 (320x320)",
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
    notes = "v4 mobile_app_training_v4. DINOv3 teacher v9 — student DEIMv2-N. " +
            "Two inputs (images, orig_target_sizes); three outputs in pixel coords. " +
            "First Pixel 5 candidate for >=10 FPS.",
)
```

Add it to `ModelRegistry.all`. Keep STUB at index 0 so the app's default
remains the safe one until the user picks the new model.

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

### 5. Add the model file to the sideload bundle

The phone app reads models from `tpl/poop_models/` (a separate
submodule). The `*.onnx` artifacts in that folder are gitignored — copy
the v4 export there manually, or via the existing
`scripts/install_to_phone.sh` flow.

```bash
cp $V4_ROOT/runs/deimv2_n_tile_g2_320x320/export/deimv2_n_h320_w320.onnx \
   $SHITSPOTTER_DPATH/tpl/poop_models/
```

### 6. Run on-device benchmark

Once sideloaded, exercise:

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

If DEIMv2-N misses the bar, fall back to DEIMv2-Pico (drop in via the
same six steps, swapping the ONNX file and the registry entry).

---

## Where the docs and lessons land

Each on-phone failure or surprise should:

* Add a new entry in
  `tpl/shitspotter-phone-app/dev/benchmark-candidates/app-deployment-questions.md`
* Append to `dev/journals/lessons_learned.md` if it's a new failure
  category (don't over-grow the file with restatements of existing
  invariants — link to the existing entry instead).
