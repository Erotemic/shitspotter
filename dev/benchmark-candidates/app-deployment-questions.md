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

---

## Compose `Modifier.transformable` silently blocks `HorizontalPager` swipe

Status: draft
Level: B
Tags: compose-gestures, pointer-event-consumption, photo-viewer, kmp-android
Requires full dataset: no
Requires trained weights: no
Pre-error SHA: e3622b1 (zoom/pan/sort + pause-on-review added)
Fix SHA:       5585432 (custom pinch/pan pointerInput replacing transformable)

### Source context

During review-screen work on the ShitSpotter KMP phone app
(`tpl/shitspotter-phone-app/`), a photo viewer was implemented using
`HorizontalPager` for swipe-between-photos and
`Modifier.graphicsLayer + Modifier.transformable` inside each pager
page for pinch-to-zoom.

Both swiping to the next page AND pinch-to-zoom stopped working after
`transformable` was added. The reported symptoms: "pinch to zoom and
swipe to move to the next photo does not work."

Key files:
- `composeApp/src/commonMain/kotlin/io/kitware/shitspotter/ui/AppScreen.kt`
  — `PhotoViewer` composable, pager page Box

### Pre-error setup

The agent adds zoom/pan to the existing `HorizontalPager` by wrapping
the page content `Box` with:

```kotlin
val transformState = rememberTransformableState { zoomChange, panChange, _ ->
    zoomScale = (zoomScale * zoomChange).coerceIn(1f, 8f)
    panOffset = if (zoomScale > 1f) panOffset + panChange else Offset.Zero
}
Box(
    modifier = Modifier
        .fillMaxSize()
        .graphicsLayer { scaleX = zoomScale; ... }
        .transformable(state = transformState),
) { Image(...) }
```

The tempting diagnosis: the detection overlay (a sibling `Box` with
`fillMaxSize`) must be consuming pointer events. Fixing the overlay
(switching from `detectTapGestures` to `awaitFirstDown(requireUnconsumed
= false)`) is plausible and worth doing for other reasons, but does NOT
restore swipe or pinch.

### Question

Given a `HorizontalPager` whose pages each carry `Modifier.transformable`
for pinch-to-zoom, both page swiping and pinch zoom stop working. The
overlay sibling's gesture handler has already been confirmed non-consuming.
Identify the true root cause and fix it so that:
1. Single-finger horizontal swipe changes pages normally.
2. Two-finger pinch changes `zoomScale` and does not trigger a page change.
3. Single-finger drag while zoomed (`zoomScale > 1`) pans the image
   without changing pages.

### Expected answer

`Modifier.transformable` wraps `detectTransformGestures`, which calls
`event.changes.forEach { it.consume() }` for **any** pointer event once
movement exceeds `touchSlop` — including single-finger horizontal drags
at `zoomScale == 1f`. Because `transformable` is on the child `Box`
inside the pager, it runs first in the Main pass and consumes the swipe
before the pager's `scrollable` ever sees it.

The fix is to **replace `transformable` with a custom `pointerInput`**:

```kotlin
.pointerInput(Unit) {
    awaitEachGesture {
        awaitFirstDown(requireUnconsumed = false)
        while (true) {
            val event = awaitPointerEvent()
            val pressed = event.changes.filter { it.pressed }
            if (pressed.isEmpty()) break
            if (pressed.size >= 2) {
                // Pinch — compute zoom from finger distance ratio
                val a = pressed[0]; val b = pressed[1]
                val curr = (a.position - b.position).getDistance()
                val prev = (a.previousPosition - b.previousPosition).getDistance()
                if (prev > 0f && curr > 0f) zoomScale = (zoomScale * curr / prev).coerceIn(1f, 8f)
                val centroidDelta = (a.position + b.position) / 2f -
                    (a.previousPosition + b.previousPosition) / 2f
                if (zoomScale > 1f) panOffset += centroidDelta
                event.changes.forEach { it.consume() }
            } else if (zoomScale > 1.05f) {
                // Pan while zoomed — single finger, consume
                val p = pressed.first()
                val delta = p.position - p.previousPosition
                if (delta.getDistance() > 0f) { panOffset += delta; p.consume() }
            }
            // Single-finger at zoom=1: no consume → pager handles swipe
        }
    }
}
```

### Invariant

`Modifier.transformable` unconditionally consumes single-finger drag
events once `touchSlop` is exceeded. It must not be placed on any
composable that is a descendant of a scrollable container (such as
`HorizontalPager`) when the scroll and the transform are intended to
coexist. Use a selective custom `pointerInput` instead.

### Validation

Write a Compose UI test (or verify manually on device) that:
1. With `zoomScale == 1f`, a horizontal fling changes `pagerState.currentPage`.
2. With two-finger pinch, `zoomScale` increases from 1f.
3. With `zoomScale > 1f`, single-finger drag changes `panOffset`.

Lightweight static check: grep for `Modifier.transformable` inside any
composable that is inside `HorizontalPager` or `LazyColumn` and flag it
as a potential gesture conflict.

### Wrong answers to reject

- "Fix the overlay's `detectTapGestures` to not consume down." That is a
  separate real bug (it also prevents swipe/pinch), but fixing it alone
  does NOT restore functionality because `transformable` consumes first.
- "Set `userScrollEnabled = false` on the pager." That disables page
  changes entirely rather than making them coexist with zoom.
- "Use `panZoomLock = true` on `transformable`." That controls rotation
  lock, not single-finger-pan consumption; the swipe still breaks.
- "Remove `graphicsLayer` and use `Modifier.scale`." Unrelated; the issue
  is event consumption, not the draw transform.

### Notes

`positionChanged()` is not available on `PointerInputChange` in
Compose Multiplatform 1.7.0 commonMain. Use
`(p.position - p.previousPosition).getDistance() > 0f` instead.

The same class of bug applies any time `detectTransformGestures` or
`transformable` is placed inside a `LazyColumn` row — it will consume
vertical scroll events and break the column's scrolling.

---

## `detectTapGestures` in a full-screen overlay consumes pointer-down, breaking all gesture pass-through

Status: draft
Level: A
Tags: compose-gestures, pointer-event-consumption, overlay, photo-viewer, kmp-android
Requires full dataset: no
Requires trained weights: no
Pre-error SHA: 81ab1b9 (detection overlay + per-box annotation added)
Fix SHA:       460146a (custom awaitEachGesture non-consuming tap handler)

### Source context

The photo viewer in `AppScreen.kt` draws a `PhotoDetectionOverlay` Box
(`Modifier.fillMaxSize()`) as a sibling of (and above) `HorizontalPager`.
The overlay used `detectTapGestures` to let users tap on detection boxes:

```kotlin
Box(modifier = Modifier.fillMaxSize().pointerInput(...) {
    detectTapGestures { offset -> /* hit test */ }
})
```

After this overlay was added, no gestures — swipe, pinch, or tap on
areas outside boxes — passed through to the pager.

### Pre-error setup

The overlay is a sibling of `HorizontalPager` in a `Box`. It is drawn
on top (higher z-order). The tempting reading: "the overlay is just a
transparent hit-test layer; it can't block the pager."

### Question

Given a `Box` overlay that fills the screen and uses `detectTapGestures`,
and a `HorizontalPager` sibling drawn below it, explain why pager swipe
gestures stop working. Then replace the overlay's gesture handler so
that only events that land on a drawn bounding box are consumed, and all
other events pass through to the pager.

### Expected answer

`detectTapGestures` is implemented as `awaitEachGesture {
awaitFirstDown(); down.consume(); ... }`. It **always consumes the
pointer-down event**, even before knowing whether the tap will land on
anything relevant. Because the overlay is higher z-order, it processes
events first in the Main pass; consuming the down means the pager's
scrollable never sees a valid gesture start.

The fix: replace `detectTapGestures` with a custom `awaitEachGesture`
loop that calls `awaitFirstDown(requireUnconsumed = false)` (no
implicit consume), then polls `awaitPointerEvent()` without consuming.
On detecting multi-touch or movement > `touchSlop`, the loop breaks and
events fall through. Only when a tap-up lands within a bounding box
does the handler call `upChange.consume()`:

```kotlin
awaitEachGesture {
    val down = awaitFirstDown(requireUnconsumed = false)
    var tapUp: PointerInputChange? = null
    loop@ while (true) {
        val event = awaitPointerEvent()
        if (event.changes.count { it.pressed } > 1) break  // pinch → bail
        for (change in event.changes) {
            if (change.id != down.id) continue
            if (!change.pressed) {
                if ((change.position - down.position).getDistance() <= slop) tapUp = change
                break@loop
            }
            if ((change.position - down.position).getDistance() > slop) break@loop
        }
    }
    val upChange = tapUp ?: return@awaitEachGesture
    // hit test → if hit: upChange.consume() + handle; else: nothing
}
```

### Invariant

A full-screen `pointerInput` overlay must not call `awaitFirstDown()`
(which consumes by default) unless it intends to own every gesture that
passes through it. If the overlay should only handle taps on specific
elements, it must use `requireUnconsumed = false` and defer consumption
until a hit is confirmed.

### Validation

Manual: with the overlay present, verify that a horizontal swipe changes
pages and a tap on empty space does nothing. A Compose UI test can
assert that `pagerState.currentPage` changes after a simulated horizontal
drag even while the overlay is active.

Static check: grep for `detectTapGestures` inside any composable that
is a sibling of or ancestor of a scrollable container and flag it.

### Wrong answers to reject

- "Remove the overlay." That removes the feature (box annotation).
- "Put `zIndex(-1f)` on the overlay." Lower z-order changes hit-test
  priority but `detectTapGestures` still consumes what it receives.
- "Use `Modifier.pointerInteropFilter`." That is Android-specific and
  does not fix the underlying consume-on-down problem.

### Notes

This bug and the `transformable` bug above are independent. Fixing only
one of them is insufficient: `detectTapGestures` blocks swipe/pinch
even when `transformable` is absent; `transformable` blocks swipe even
when the overlay is absent. Both must be fixed simultaneously to restore
full gesture behaviour.

---

## Android `PrintlnLogger` output is silenced by `-s` logcat filter

Status: draft
Level: A
Tags: android-logging, debugging, kmp-android, logcat
Requires full dataset: no
Requires trained weights: no
Pre-error SHA: 049a47d (AndroidLogger introduced)
Fix SHA:       12d3328 (return-type fix, same change)

### Source context

The ShitSpotter KMP app's shared `PrintlnLogger` routes all log output
through Kotlin `println()`. On Android, `println()` is captured by the
Android runtime as `System.out` logcat entries. The `install_to_phone.sh`
script uses `adb logcat -s ShitSpotter.*:V AndroidRuntime:E ...` which
silences `System.out` entirely. Running the app with `scripts/install_to_phone.sh run`
produced no log output at all, making any runtime debugging impossible.

### Pre-error setup

`PrintlnLogger` was intentionally kept in `commonMain` so it works on
desktop too. The Android callers (`MainActivity`, `AndroidBackendManager`,
`CameraAnalysisLoop`) imported `PrintlnLogger` directly. The logcat
filter in the script appeared correct because it listed `ShitSpotter.*`,
but no code ever called `android.util.Log`.

### Question

Given a KMP Android app that uses a shared `PrintlnLogger` (which calls
`kotlin.io.println`) and a logcat tail filter of `-s ShitSpotter.*:V`,
explain why no app logs appear. Add an Android-specific logger that
routes through `android.util.Log` using `ShitSpotter.<tag>` log tags,
and wire it into all Android-side callers without modifying the shared
`commonMain` `PrintlnLogger`.

### Expected answer

Create `composeApp/src/androidMain/.../AndroidLogger.kt`:

```kotlin
object AndroidLogger : AppLogger {
    override fun info(tag: String, msg: String) { Log.i("ShitSpotter.$tag", msg) }
    override fun warn(tag: String, msg: String, t: Throwable?) { ... }
    override fun error(tag: String, msg: String, t: Throwable?) { ... }
}
```

Note: `Log.i()` returns `Int`. Using `= Log.i(...)` as an expression body
causes a compile error ("return type not a subtype of Unit"). Use block
body `{ Log.i(...) }` instead.

Replace `PrintlnLogger` with `AndroidLogger` in all `androidMain`
callers. No import is needed when the logger and callers are in the same
`io.kitware.shitspotter.android` package.

### Invariant

Any log calls intended to be captured by a `adb logcat -s <tag>:V`
filter must go through `android.util.Log.X("tag", msg)` — not through
`println()`, `System.out`, or any JVM stream that Android captures under
the `System.out` logcat tag.

### Validation

```bash
# After installing, with app running:
adb logcat -s "ShitSpotter.MainActivity:V" -d | grep -c "photo saved"
# Should be > 0 after taking a photo
```

### Wrong answers to reject

- Add `System.out` to the `-s` filter list. That works but defeats the
  purpose of the filter (too noisy in production).
- Modify `PrintlnLogger` in `commonMain` to call `android.util.Log`.
  `android.util.Log` is not available in `commonMain` and would break
  the desktop target.
- Use `actual`/`expect` on `PrintlnLogger`. Overkill; a separate
  `AndroidLogger` object is simpler and the `AppLogger` interface already
  provides the abstraction point.
