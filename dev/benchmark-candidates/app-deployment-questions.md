# Benchmark candidates — app deployment

Hard-problem invariants discovered during the v2 phone-app scaffold
(2026-05-10). Each entry follows the format from
[`../AGENT_BENCHMARK_DISCIPLINE.md`](../AGENT_BENCHMARK_DISCIPLINE.md).

## How to reproduce the pre-error state

Every candidate names a **"pre-error SHA"** and a **"fix SHA"**. To
recreate the bug in a worktree:

```bash
# Land on the pre-error commit (read-only worktree, leaves main alone):
git worktree add /tmp/ssp-bench <pre-error-sha>
cd /tmp/ssp-bench/tpl/shitspotter-phone-app
source /data/tmp/shitspotter-app-toolchain/env.sh

# Inspect the diff that fixed it:
git -C /home/joncrall/code/shitspotter show <fix-sha> -- tpl/shitspotter-phone-app/

# After investigating, clean up:
git worktree remove /tmp/ssp-bench
```

All SHAs in this file are short (7-char) refs into the `main` branch
of `Erotemic/shitspotter` as of 2026-05-10.

---

## ONNX shape mismatch fails at session.run, not at load

Status: draft
Level: A
Tags: preprocessing-parity, onnx-export-parity, model-spec, fail-fast
Requires full dataset: no
Requires trained weights: yes (just the small YOLOX-nano poop ONNX)
Pre-error SHA: 92edfe4 (real-model end-to-end + backend comparison harness)
Fix SHA:       21a3b43 (ONNX shape-mismatch validation + describe-model CLI)

### Source context

While scaffolding `tpl/shitspotter-phone-app/`, I wired
`OnnxRuntimeJvmBackend(ModelSpec, onnxPath)` where the `ModelSpec`
declared `inputWidth=640, inputHeight=640` (it inherited from a
default-stub spec) but the actual ONNX file was a 416x416 YOLOX-nano.
The Kotlin code accepted the spec, constructed the session
successfully, and then crashed deep inside `session.run` with:

```text
ai.onnxruntime.OrtException: Error code - ORT_INVALID_ARGUMENT
 - message: Got invalid dimensions for input: images …
 index: 2 Got: 640 Expected: 416
 index: 3 Got: 640 Expected: 416
```

Files involved:
- `composeApp/src/desktopMain/.../OnnxRuntimeJvmBackend.kt`
- `composeApp/src/androidMain/.../OnnxRuntimeAndroidBackend.kt`
- `composeApp/src/commonMain/.../ModelSpec.kt`

### Pre-error setup

The agent has a `ModelRegistry` with several `ModelSpec` entries and a
freshly added ONNX file. The tempting wrong path is to load the ONNX,
call `session.run` once, and treat any first-frame failure as a runtime
condition. ORT delays shape validation until first run, so the
construction path *looks* clean.

A second misleading nearby fact: `session.inputNames` works fine. The
backend dev only sees the failure once a frame actually flows.

### Question

Given an ONNX backend that wraps `ai.onnxruntime.OrtSession`, design
the construction-time validation that catches `ModelSpec` /
`InputInfo.shape` mismatches before the first `analyze()` call, and
write a smoke test that asserts the wrong spec is rejected without
having to feed a frame.

### Expected answer

After `env.createSession(...)`, read
`session.inputInfo[name].info as TensorInfo` (the cast is required —
non-tensor inputs have to be skipped, not crashed on). For a 4D shape
in `[N, C, H, W]` (NCHW) or `[N, H, W, C]` (NHWC) form, pick out the
H and W indices based on `spec.inputLayout` and assert that
`shape[h] == spec.inputHeight && shape[w] == spec.inputWidth`. Dynamic
dims (the value `-1`) must be allowed through.

The error message must name the `ModelSpec.modelId` so the operator
knows which registry entry to fix, not just "ORT failed".

### Invariant

The construction of a `DetectorBackend` for a real ONNX file must
validate the model's declared input H/W against the supplied
`ModelSpec` and fail at construction with a Kotlin
`IllegalStateException` referencing the `modelId`. It must accept
dynamic dimensions and must not mutate the model.

### Validation

```text
./gradlew :composeApp:desktopTest --tests \
    "io.kitware.shitspotter.desktop.OnnxShapeValidationTest"
```

The test loads the real YOLOX-nano file with a deliberately wrong
640x640 spec and asserts `OnnxRuntimeJvmBackend(...)` throws. Skips
cleanly if the model file isn't on disk.

### Wrong answers to reject

- Catch and re-throw the `OrtException` from inside `analyze`; the
  failure surfaces too late and obscures which spec is wrong.
- Only validate width or only validate height; both H and W can drift.
- Hardcode `4D` and crash on non-tensor inputs; some ONNX models use
  sequence or map inputs.
- Treat dynamic dims (`-1`) as a validation failure.

### Notes

`AndroidShapeValidationTest` is not yet wired (the
`androidInstrumentedTest/` source set has a placeholder only).
Mirror this Kotlin pattern when adding it.

---

## PreviewView FILL_CENTER vs overlay FIT_CENTER produces misaligned boxes

Status: draft
Level: B
Tags: phone-deployment, overlay-coordinate-system, scale-type-parity
Requires full dataset: no
Requires trained weights: no
Pre-error SHA: 40197ac (Milestone 0 stack decision + KMP+Compose scaffold —
               DetectionOverlay defaulted to min-scale letterbox)
Fix SHA:       1eba4b8 (align DetectionOverlay scaling with PreviewView FILL_CENTER)

### Source context

CameraX `PreviewView` defaults to `ScaleType.FILL_CENTER` — scale to
fill, crop excess. My initial `DetectionOverlay` Composable computed
its scale as `min(scaleX, scaleY)` (FIT_CENTER, the letterbox case),
which is fine in isolation but produced two coordinate spaces over
the same `Box`. On a 4:3 frame on a 16:9 phone the boxes rendered in
a different region of the screen than the camera image.

Files:
- `composeApp/src/commonMain/kotlin/io/kitware/shitspotter/ui/Overlay.kt`
- `composeApp/src/commonMain/kotlin/io/kitware/shitspotter/ui/AppScreen.kt`
- `composeApp/src/androidMain/.../AndroidCameraSurface.kt`
- `composeApp/src/desktopMain/.../DesktopCameraSurface.kt`

### Pre-error setup

The agent sees a working CameraX preview and a working YOLOX
postprocess. The tempting wrong path: assume "scaleX/scaleY are
equal because aspect ratios match" and pick whichever scale math
feels cleaner. The agent doesn't know the desktop harness has a
*different* preview-render path (it draws a still image with
min-scale letterbox) until they look at both.

### Question

Given a Compose Multiplatform overlay that draws detection boxes over
a camera-or-still-image surface, design the API so each platform's
surface can declare its own scale type and the overlay automatically
matches. Demonstrate that the same overlay aligns correctly on Android
(`PreviewView.FILL_CENTER`) and on desktop (still-image
`FIT_CENTER`).

### Expected answer

Add an `OverlayScaleMode` enum (`FIT_CENTER` / `FILL_CENTER`) and a
default-valued property `CameraSurface.overlayScaleMode`. The shared
`AppScreen` reads it and forwards to `DetectionOverlay(scaleMode = …)`.
`AndroidCameraSurface` defaults to `FILL_CENTER`,
`DesktopCameraSurface` overrides to `FIT_CENTER`. The overlay picks
`maxOf(scaleX, scaleY)` for FILL_CENTER and `minOf(...)` for
FIT_CENTER. Centre offset is computed the same way in both cases.

### Invariant

Any `CameraSurface` implementation must declare its preview scale type,
and the `DetectionOverlay` must use that declaration. The shared UI
code must not assume one scale type for all platforms.

### Validation

Static / structural: search the codebase for direct
`DetectionOverlay(scaleMode = …)` usages and verify each `CameraSurface`
implementation overrides `overlayScaleMode`.

```bash
grep -r "DetectionOverlay\|overlayScaleMode" \
    tpl/shitspotter-phone-app/composeApp/src/
```

End-to-end visual: run `./gradlew :composeApp:run --args="--image=…"`
and confirm the desktop overlay box aligns with the displayed image.
Pixel 5 alignment validation requires a phone — flag in the report.

### Wrong answers to reject

- Hard-code FILL_CENTER everywhere; the desktop still-image surface
  uses FIT_CENTER deliberately so detection boxes don't render in a
  cropped-out region of the canvas.
- Compute scale from `state.lastFrameWidth/Height` vs canvas size in
  the overlay without consulting the surface. The frame may be
  rotated relative to the preview; that's a *separate* fix
  (`BoundingBox.rotated`), not solvable here.

### Notes

Pairs with `BoundingBox.rotated(degrees, w, h)` which solves the
related-but-distinct rotation-orientation issue.

---

## Camera rotationDegrees is non-zero on portrait-locked Android phones

Status: draft
Level: B
Tags: phone-deployment, exif-orientation, coordinate-transform
Requires full dataset: no
Requires trained weights: no
Pre-error SHA: e82ccc3 (Milestone 2 ONNX backends + Android APK builds —
               CameraAnalysisLoop ignored ImageProxy.rotationDegrees)
Fix SHA:       feca293 (rotate detection boxes into display orientation)

### Source context

CameraX delivers `ImageProxy` buffers in their native sensor
orientation. On Pixel 5 in portrait-locked mode, that's a 480x640
landscape buffer with `imageInfo.rotationDegrees = 90`. PreviewView
rotates the preview internally; the detector receives the un-rotated
buffer and emits detections in *buffer* coordinates. The overlay,
running over PreviewView output, sees the rotated preview but the
buffer-space coordinates — so boxes render in the wrong place.

Files:
- `composeApp/src/androidMain/.../CameraAnalysisLoop.kt`
- `composeApp/src/commonMain/.../Geometry.kt::BoundingBox.rotated`
- `composeApp/src/androidMain/.../ImageProxyFrame.kt`

### Pre-error setup

The agent has the model producing detections that round-trip through
the YOLOX postprocess correctly. Visual smoke test on a phone: the
boxes appear in a vertically-flipped or horizontally-flipped region
of the screen relative to the actual subject. The tempting wrong
path: rotate the camera preview to match the buffer (don't — the
preview is already correctly rotated by PreviewView). The other
tempting wrong path: rotate the canvas in `DetectionOverlay` (don't
— the canvas is in display space; the boxes need to *enter* display
space).

### Question

Given that `CameraAnalysisLoop.handleFrame` receives an
`ImageProxyFrame` with `rotationDegrees ∈ {0, 90, 180, 270}` and the
detector produces boxes in buffer-space pixels, design the
transformation to display-space pixels that the `DetectionOverlay`
can consume directly, including the rotated frame W/H so the
overlay's scale math uses the right aspect ratio.

### Expected answer

Add a pure-Kotlin `BoundingBox.rotated(degrees, frameW, frameH)` that
returns `Triple<rotatedBox, newFrameW, newFrameH>`. The math is the
standard 2D rotation around the frame centre with W/H swap for
90/270:

```text
deg 0:   identity
deg 90:  new x = frameH - (y+h); new y = x; new w = h; new h = w
                ; newFrameW = frameH; newFrameH = frameW
deg 180: new x = frameW - (x+w); new y = frameH - (y+h)
deg 270: new x = y; new y = frameW - (x+w); same swap
```

In `CameraAnalysisLoop.handleFrame`, after the score-threshold filter,
map each detection's box through `rotated(frame.rotationDegrees, …)`
and push the **rotated frame W/H** into `AppState`. Use a dummy
zero-sized box to compute the new frame dimensions when the detection
list is empty.

### Invariant

The (rotated frame W, rotated frame H, rotated detection boxes) tuple
in `AppState` must be self-consistent — they must all describe the
same display-space coordinate system.

### Validation

Pure-Kotlin unit tests in `commonTest`:

- identity (deg 0)
- 90 ↔ 270 swap axes correctly
- 180 inverts origin
- normalisation (-90 == 270, 450 == 90)
- four quarter-rotations return to original
- double-rotation at 180 returns to original
- only multiples of 90 are accepted

```bash
./gradlew :composeApp:desktopTest --tests \
    "io.kitware.shitspotter.core.BoundingBoxRotationTest"
```

Visual / on-device validation requires a Pixel 5.

### Wrong answers to reject

- Rotating only the box without updating the frame dimensions —
  `DetectionOverlay` scales by `size / frame`, so a wrong frame size
  produces wrong scale.
- Using `imageInfo.rotationDegrees` as the rotation passed to
  `Bitmap.createBitmap`'s matrix without accounting for the +y axis
  difference between Bitmap and Compose coordinate spaces (this *is*
  done deliberately in `ImageProxyFrame.encodeJpeg` to produce a JPEG
  matching the displayed orientation; do not generalise that helper).
- Skipping the rotation on the empty-detections path — the frame
  dimensions in `AppState` would diverge from the overlay's
  assumption on the very next non-empty frame.

### Notes

The implementation lives in
`Geometry.kt::BoundingBox.rotated`. The same math will be needed for
the eventual Compose UI test that validates the overlay end-to-end
on a synthetic 90°-rotated frame.

---

## YOLOX-nano-poop is a cropped-only model and over-fires on full images

Status: draft
Level: A
Tags: phone-deployment, model-export, ood-behaviour, preprocessing-parity
Requires full dataset: no
Requires trained weights: yes
Pre-error SHA: 92edfe4 (real-model end-to-end + backend comparison harness —
               model was wired before the OOD note was added)
Fix SHA:       fe1b1c8 (docs/004_kotlin_python_parity.md captures the
               degenerate-box behaviour; ModelSpec.notes updated in eb85008)

### Source context

The first real model wired into the v2 app is
`tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx`. Run against
`tpl/YOLOX/assets/dog.jpg` it returns 8 above-threshold detections in
the Python reference and 3 after Kotlin NMS, with the top score at
0.757. Inspecting the boxes shows they are degenerate (~2×3 px,
located off the top edge of the source image).

Files:
- `tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx`
- `tpl/shitspotter-phone-app/docs/004_kotlin_python_parity.md`
- `tpl/shitspotter-phone-app/composeApp/src/commonMain/.../ModelSpec.kt::YOLOX_NANO_POOP`

### Pre-error setup

The agent runs the model against a non-poop image during scaffolding,
sees high-confidence detections, and is tempted to either (a) lower
the score threshold (wrong — the high-confidence ones are the
problem) or (b) blame the NMS / postprocess / preprocessing for the
discrepancy with intuition. The tempting wrong path here is to "fix"
the postprocess to suppress these.

### Question

The YOLOX-nano poop model reports a 0.757-score detection on a
stock dog photo with a 2×3-pixel box at the top of the frame. Identify
why this is *not* a bug in the Kotlin / Python preprocess, postprocess,
or NMS, and document the model behaviour without changing the
detection pipeline.

### Expected answer

The model was trained on **pre-cropped patches**, not full-image
scenes. Out-of-distribution full-image inputs produce high-confidence
garbage. The detection pipeline (letterbox → YOLOX decode → NMS) is
correct; what's wrong is the use-case — the model should only be
invoked on user-aimed crops (the original MAUI app pattern), or be
replaced with a full-image-trained model.

The right action is documentation, not code:
- `ModelSpec.YOLOX_NANO_POOP.notes` records the cropped-only origin.
- `docs/004_kotlin_python_parity.md` documents the dog.jpg over-fire
  so it doesn't get reported as a regression on the next round of
  benchmarks.

### Invariant

Every `ModelSpec` for a real-world model must carry a `notes` field
recording any known out-of-distribution behaviour, training data
distribution, and known-bad-on-X cases.

### Validation

Pure-Kotlin sanity test: `Yolox.postprocessDecoded` filters and NMS
exactly as designed.

```bash
./gradlew :composeApp:desktopTest --tests \
    "io.kitware.shitspotter.core.YoloxPostprocessExtraTest"
```

Parity test: `scripts/python_reference_compare.py` against the same
image produces the same top-3 score within ~0.01 (Pillow vs
java.awt.image precision drift).

### Wrong answers to reject

- Drop the score threshold lower — makes the problem worse.
- Raise the NMS IoU threshold — the degenerate boxes don't overlap
  and won't be suppressed.
- "Fix" the YOLOX postprocess — it's not wrong.
- Quietly clamp boxes to the visible image area — hides the
  underlying model issue.

### Notes

The right long-term fix is a full-image-trained replacement model.
That work lives in a separate workstream; the app-side
contribution is to make swap-in painless via the `ModelRegistry`.

---

## Default APK ABI filters bundle 4x the native libraries

Status: draft
Level: A
Tags: phone-deployment, packaging, native-libraries, release-engineering
Requires full dataset: no
Requires trained weights: no
Pre-error SHA: e82ccc3 (Milestone 2 ONNX backends + Android APK builds —
               first 82 MB debug APK landed here)
Fix SHA:       8e4f892 (29 MB arm64-v8a APK + raw-strides decode tests)

### Source context

The first `assembleDebug` produced an 82 MB APK. Inspecting the APK
contents revealed `lib/armeabi-v7a/`, `lib/arm64-v8a/`, `lib/x86/`,
and `lib/x86_64/` each containing a ~20 MB copy of
`libonnxruntime.so` plus CameraX native libs. The Pixel 5 is arm64;
the other three ABIs are dead weight on the device.

Files:
- `tpl/shitspotter-phone-app/composeApp/build.gradle.kts`

### Pre-error setup

The agent assembles a debug APK and notes the 82 MB size as
"surprising but not blocking". The tempting wrong path is to inspect
DEX size or proguard rules first; the actual culprit is in
AGP defaults for native library packaging.

### Question

The Android debug APK for an ONNX-Runtime-using project is 82 MB on
disk. Identify the largest contributor and produce a Gradle config
change that shrinks the install for a single-ABI target (Pixel 5 =
arm64-v8a) without losing inference capability on that device.

### Expected answer

Add `ndk.abiFilters = setOf("arm64-v8a")` to
`composeApp/build.gradle.kts`'s `android.defaultConfig`. This drops
the APK from 82 MB to 29 MB by excluding `armeabi-v7a`, `x86`, and
`x86_64` native libs. Document the trade-off (need to add x86_64 if
emulator-testing, armeabi-v7a for pre-arm64 devices) in a comment so
the next agent doesn't drop the filter accidentally.

### Invariant

The release APK must declare its supported ABIs explicitly. Defaulting
to "every ABI" is a packaging bug, not a feature; for a known target
(Pixel 5) the filter should be set.

### Validation

```bash
./gradlew :composeApp:assembleDebug
du -h composeApp/build/outputs/apk/debug/composeApp-debug.apk
# expect ~29 MB, not ~80 MB
```

Inspect the APK to confirm only arm64-v8a is bundled:

```bash
unzip -l composeApp/build/outputs/apk/debug/composeApp-debug.apk | \
    grep "^.*lib/"
```

### Wrong answers to reject

- Switch to ProGuard / R8 first; minification doesn't strip native
  `.so` files for other ABIs.
- Adopt App Bundle (`.aab`) without setting the filter; AAB *would*
  let the Play Store deliver only arm64 to the user, but our scaffold
  emits APKs for sideloading, so the filter still matters.
- Strip the ONNX Runtime dependency; we still need it on Pixel 5.

### Notes

If a future agent re-adds an emulator workflow, set the filter to
`setOf("arm64-v8a", "x86_64")` rather than removing it; emulators on
x86 hosts need x86_64. Document the choice in the same comment.
