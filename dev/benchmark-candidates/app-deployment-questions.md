# Benchmark candidates — app deployment

Hard-problem invariants discovered during the v2 phone-app scaffold
that future agents are likely to re-discover the wrong way. Each
entry is a question + the failure mode + the fix.

## 1. Model input shape mismatch fails inside `session.run`, not at load

**Question:** Why did "loading a YOLOX-nano-poop model with the stub
ModelSpec" produce a 5-line ORT stack trace deep inside CameraX
analysis instead of failing at app startup?

**Failure mode:** ONNX Runtime defers shape validation until first
`run`. A wrong-spec backend constructs successfully and you only learn
about the mismatch when you actually feed it a frame. Kotlin's
exception comes from native code with low context.

**Fix:** Read `session.inputInfo[name].info.shape` after creating the
session and compare against the `ModelSpec`. Throw a Kotlin
`IllegalStateException` with the modelId pointing at the right config
file. Implemented in
`tpl/shitspotter-phone-app/composeApp/src/desktopMain/kotlin/io/kitware/shitspotter/desktop/OnnxRuntimeJvmBackend.kt::validateInputShape`
and the Android twin.

**Benchmark candidate:** "every backend exposes an
input/output-shape-mismatch test that fails at construction, not at
first analyze." Lock with `OnnxShapeValidationTest`.

## 2. PreviewView FILL_CENTER + DetectionOverlay FIT_CENTER = misaligned boxes

**Question:** Why did the overlay boxes appear in a different region
of the screen than the camera image even though the math was correct?

**Failure mode:** PreviewView defaults to FILL_CENTER (scale to fill,
crop excess); DetectionOverlay defaulted to a min-scale letterbox
(scale to fit, leave padding). Two coordinate spaces, both reasonable
in isolation, but rendered into the same Box.

**Fix:** Have CameraSurface declare its scale type and have the
overlay match. Implemented as
`CameraSurface.overlayScaleMode` + `DetectionOverlay(scaleMode = …)`.
Desktop's still-image surface declares FIT_CENTER, Android's CameraX
surface declares FILL_CENTER.

**Benchmark candidate:** "every CameraSurface implementation must
declare its `overlayScaleMode` and the overlay must read it."

## 3. Camera rotationDegrees is ignored by default

**Question:** Why did the boxes look right in landscape but not
portrait?

**Failure mode:** CameraX delivers buffers in their native
landscape orientation with rotationDegrees telling you how to rotate
to natural. PreviewView rotates the preview internally; the overlay
did not until very late in the scaffold.

**Fix:** Rotate detection boxes via `BoundingBox.rotated(degrees, w, h)`
in `CameraAnalysisLoop` before pushing into `AppState`. Pure-Kotlin
function so it lives in commonMain and is unit-tested by
`BoundingBoxRotationTest`.

**Benchmark candidate:** "frame.rotationDegrees > 0 produces
visibly aligned boxes" — hard to assert from a test, easy to
regress. Pixel 5 portrait-locked validation should specifically
report on this.

## 4. YOLOX-nano-poop is a cropped-only model and over-fires on full images

**Question:** Why does the model give a 0.757 confidence "poop"
detection on a stock dog photo?

**Failure mode:** The shipped YOLOX-nano model is *cropped-only* —
it was trained on pre-cropped patches, not full-image scenes. Out-of-
distribution images produce high-confidence garbage that fails NMS
because the boxes are tiny and don't overlap.

**Fix:** Document this in
`tpl/shitspotter-phone-app/docs/004_kotlin_python_parity.md` so it's
not reported as a regression. The proper fix is a full-image model;
the stop-gap is to train downstream filtering or only run inference
when the user explicitly aims at the ground.

**Benchmark candidate:** "any new model spec includes an OOD
behaviour note" — covered by the Notes field in `ModelSpec` and the
template in `docs/006_adding_a_new_model.md`.

## 5. APK ABI defaults double the install size

**Question:** Why is the debug APK 82 MB?

**Failure mode:** AGP's default packaging includes
`armeabi-v7a + arm64-v8a + x86 + x86_64` for any `*.so` from a
dependency — and ONNX Runtime ships native libs for all four. On a
single arm64 phone that is 50 MB of dead weight.

**Fix:** Set `ndk.abiFilters = ["arm64-v8a"]` in the Android
`defaultConfig`. Knocks the APK from 82 MB to 29 MB without losing
arm64 capabilities.

**Benchmark candidate:** "if a release-mode APK is over 30 MB, check
the ABI filter first." Locked into `composeApp/build.gradle.kts` with
a comment pointing at the trade-off.
