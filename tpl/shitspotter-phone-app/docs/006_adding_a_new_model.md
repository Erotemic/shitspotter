# 006 — Adding a new detector model

A practical recipe for plugging a new detector into the phone-app
without touching code paths that already work. Walk through the four
steps in order.

## Step 1 — Inspect the ONNX file

Always confirm the actual input/output shape and dtype before
guessing. The desktop CLI does this in one line:

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
cd tpl/shitspotter-phone-app
./gradlew :composeApp:run --console=plain --args="describe \
   --model=/path/to/your_model.onnx"
```

Expected output looks like:

```text
inputs:
  images : shape=[1, 3, 416, 416] type=FLOAT
outputs:
  output : shape=[1, 3549, 6] type=FLOAT
```

Decode that:

- `[1, 3, H, W]` → NCHW with 3-channel input, height H, width W.
- `[1, 3549, 6]` → 3549 anchors of `(cx, cy, w, h, obj, cls0)`.
  YOLOX postprocess applies directly.

If your output looks like `[1, num_classes+5, num_anchors]`, the
postprocess will need to transpose first — file an issue and follow
the same recipe as `Yolox.decodeRawStrides`.

## Step 2 — Add a `ModelSpec` entry

In `composeApp/src/commonMain/kotlin/io/github/erotemic/shitspotter/core/ModelSpec.kt`,
add a constant inside `ModelSpec.Companion`:

```kotlin
val YOUR_MODEL = ModelSpec(
    modelId = "your-model-v1",                // unique id; used by registry + UI
    displayName = "Your model (640x640)",     // shown in the chip row
    modelFile = "your_model.onnx",            // basename only; lookup is by name
    format = ModelFormat.ONNX,                // ONNX, TFLITE (stub), PTE, COREML, STUB
    inputWidth = 640,
    inputHeight = 640,
    inputLayout = InputLayout.NCHW,
    colorOrder = ColorOrder.RGB,
    normalization = Normalization(scale = 1f / 255f),  // or scale=1f for YOLOX
    resizePolicy = ResizePolicy.LETTERBOX,    // STRETCH or CENTER_CROP also work
    postprocessType = PostprocessType.YOLOX,
    classNames = listOf("poop"),
    scoreThreshold = 0.25f,
    iouThreshold = 0.45f,
    notes = "Where the weights came from + any quirks worth recording.",
)
```

Then add it to `ModelRegistry.all`:

```kotlin
val all: List<ModelSpec> = listOf(
    ModelSpec.STUB,
    ModelSpec.YOLOX_NANO_POOP,
    ModelSpec.CUSTOM_V5_EPOCH115,
    ModelSpec.CUSTOM_V2_EPOCH126,
    ModelSpec.YOUR_MODEL,                      // ← here
)
```

That's it for code. The chip row, threshold slider, settings store,
failure-case metadata, and CompareCli all pick up the new id by virtue
of being registered.

## Step 3 — Tell the runtime where the file is

There's no commit step here — the rule is "weights stay out of git".
Pick whichever placement matches your workflow:

- **Sideload (easiest)**: add the filename to `KNOWN` in `scripts/push_models.sh` and run it.
  That script searches three locations automatically (see below) and pushes to the right
  on-device path for debug or release.
- **Manual adb push**:
  ```bash
  # debug build
  adb push your_model.onnx /sdcard/Android/data/io.github.erotemic.shitspotter.debug/files/models/
  # release build
  adb push your_model.onnx /sdcard/Android/data/io.github.erotemic.shitspotter/files/models/
  ```
- **Bundled in APK**: `cp your_model.onnx composeApp/src/androidMain/assets/your_model.onnx`
  (gitignored via `*.onnx` rule — only do this for small default models)
- **Desktop one-off**: `--model=/path/to/your_model.onnx --model-id=your-model-v1`

`AndroidModelLoader` falls through external-files-dir → cache → assets
in that order, so any of the above will work.

### Where the weights live on the workstation

`scripts/push_models.sh` searches these locations in order:

| Location | Contents |
|----------|----------|
| `tpl/poop_models/` | YOLOX-nano, custom-v5, custom-v2 |
| `/data/joncrall/shitspotter_v4/runs/*/export/` | DEIMv2 pico/n at 320/416/512/640 |
| `/data/joncrall/dvc-repos/shitspotter_dvc/models/` | YOLOv9 simple-v3 (long checkpoint filename matched by prefix) |

The script pushes all 8 registered models in one shot and skips files it can't find
locally. After pushing, switch models in the app or restart it — the loader checks
the external dir fresh on every `setActive` call, no caching of misses.

## Step 4 — Verify before declaring victory

Three checks, in order of cost:

1. **Desktop CLI smoke test** — runs against a synthetic frame, no
   ground truth needed. If this fails, the model spec is wrong.

   ```bash
   ./gradlew :composeApp:run --console=plain --args="compare \
      --image=tpl/YOLOX/assets/dog.jpg \
      --model=/path/to/your_model.onnx \
      --model-id=your-model-v1 \
      --runs=3 --warmup=1"
   ```

2. **Python parity check** — confirms the postprocess agrees with a
   minimal reference implementation.

   ```bash
   /tmp/onnx_venv/bin/python scripts/python_reference_compare.py \
      --image tpl/YOLOX/assets/dog.jpg \
      --model /path/to/your_model.onnx \
      --input-size 640 --num-classes 1 --threshold 0.25 --top 3
   ```

   If the top-N detections diverge by more than a few percent, the
   ModelSpec normalisation / colour order / resize policy is wrong.

3. **Pixel 5 sideload** — the only way to know real FPS / NNAPI
   delegate behaviour. Use `scripts/install_to_phone.sh` and report
   the HUD readings.

## Optional — add an integration test

If the model is canonical enough to live in the public registry,
mirror the existing `OnnxBackendSmokeTest` pattern with a
`<YourModel>BackendSmokeTest` that conditionally loads the file
when present and skips cleanly when absent. Drop it under
`composeApp/src/desktopTest/`.

## What can go wrong

- Input shape mismatch → `OnnxRuntimeJvmBackend.validateInputShape`
  rejects the spec at construction, before the first analyze call.
- Output shape mismatch → `Yolox.postprocessDecoded` throws
  `predictions size … not divisible by perRow`. Re-run `describe`
  and double-check the ModelSpec's classNames count.
- Color-order mismatch → silent — the model returns mostly-zero
  detections at any threshold. Verify by trying both
  `ColorOrder.RGB` and `ColorOrder.BGR` and seeing which behaves.
- Normalisation mismatch → also silent. YOLOX-style models expect
  `scale = 1f` (no /255), most other ONNX exports expect
  `scale = 1f / 255f`. The Python parity script is the fastest way
  to disambiguate.
