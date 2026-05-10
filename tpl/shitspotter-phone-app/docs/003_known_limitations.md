# 003 — Known limitations and next-agent backlog

These are deliberate non-issues for the current scaffold but real concerns
for any agent who picks up the prototype next. They are not breakages.
They are the things the next pull request should address before the app
goes anywhere near a user-facing release.

> **Update log.** Items marked **DONE** below have been resolved since
> this file was first written. Items still open are not yet implemented
> at the latest commit on `main`. Each pair has its fix SHA so an agent
> can audit instead of trusting the marker.

## Hot-path correctness

### 1. Overlay rotation handling — **DONE** (feca293, 1eba4b8)

CameraX `ImageProxy` exposes `rotationDegrees`. `CameraAnalysisLoop`
now rotates every detection box and the reported frame W/H via
`BoundingBox.rotated(degrees, w, h)` before pushing into `AppState`.
`DetectionOverlay` also reads the surface's `overlayScaleMode`
(FILL_CENTER for the Android `PreviewView`, FIT_CENTER for the
desktop still-image surface) so boxes line up with the displayed
camera image. See `BoundingBoxRotationTest` and `docs/005` §"Hot path".

### 2. JPEG capture on rotated frames

`ImageProxyFrame.encodeJpeg` does honour `rotationDegrees` (rotates the
ARGB bitmap before encoding), so the saved JPEG matches what the user
saw. But the failure-case `metadata.json` does **not** record the
rotation that was applied, which makes downstream training-set reuse
slightly ambiguous if you do something unusual at capture time. Consider
adding `rotationDegreesAtCapture: Int` to `FailureCaseMetadata`.

### 3. ONNX-output schema assumption

`OnnxRuntimeAndroidBackend` and `OnnxRuntimeJvmBackend` both assume the
model emits a single tensor of shape `[1, num_anchors, 5+num_classes]`
in input-image pixel space (decoded YOLOX). This holds for the
YOLOX-nano poop ONNX file we ship against. If you load a model whose
export embeds the per-stride raw output instead, the postprocess will
crash with `output size not divisible by perRow`. `Yolox.decodeRawStrides`
exists for that case but is not yet wired into the backends.

Fix sketch: add a `requiresStrideDecode: Boolean` to `ModelSpec` and
branch in the backends.

### 4. Preprocessing cost on Android

The `Preprocessing.letterboxRgb` + `toFloatTensor` path is pure-Kotlin
and copies bytes twice (RGBA → RGB → letterboxed → float tensor). The
desktop smoke test shows ~30 ms of preprocessing on JVM for a 480x640
input. On Pixel 5 this is likely 50-100 ms — not catastrophic but the
biggest non-inference cost. Two reasonable paths:

- Use `RenderScript` / `RenderEffect` / `Vulkan compute` for the YUV →
  RGB → tensor step.
- Use ONNX Runtime's built-in input transformations (it can accept a
  byte tensor and let the model do the float conversion internally).

This is exactly the kind of decision that wants a benchmark before
committing — see `docs/002_benchmarks_template.md`.

## Build / packaging

### 5. Release APK signing — partially done (5f209d4)

`assembleRelease` now produces a 20 MB shrunk APK with R8 minification
and resource shrinking enabled. **Still open:** the release build reuses
the debug signing config — fine for sideloading, **not** suitable for
Play Store. Replace `signingConfig = signingConfigs.getByName("debug")`
in `composeApp/build.gradle.kts` with a real keystore + secret-managed
credentials before any public release.

### 6. APK size (~82 MB)

Most of the size is the bundled ONNX Runtime native libraries
(`libonnxruntime.so` for arm64-v8a + armeabi-v7a + x86_64). To shrink
for sideload:

- Add `ndk.abiFilters = setOf("arm64-v8a")` to the Android `defaultConfig`
  once you only target real Pixel 5 / Pixel 6+ builds.
- Switch to ONNX Runtime Mobile (smaller package), at the cost of fewer
  ops.
- Strip the desktop AppImage / Deb formats from `compose.desktop` if you
  never plan to ship a Linux package.

### 7. iOS source set is unbuilt

`composeApp/src/iosMain/kotlin/io/kitware/shitspotter/core/IosActuals.kt`
exists with placeholder `actual fun nowMonoMs()` and `actual object
BuildInfo`, but `kotlin { iosArm64() }` etc. is **not** declared in
`composeApp/build.gradle.kts`. To build for iOS:

- Add `iosX64()`, `iosArm64()`, `iosSimulatorArm64()` targets.
- Run `./gradlew :composeApp:linkDebugFrameworkIosArm64` on a macOS host.
- Wire AVFoundation + Core ML / ORT-CoreML EP in `iosMain/`.
- Sign + archive via Xcode.

None of this is testable from the Linux VM.

## Coverage / testing

### 8. Android instrumented tests are not wired

`composeApp/src/androidInstrumentedTest/` doesn't exist. To validate the
camera analysis loop end-to-end we'd want an instrumented test using
`androidx.test.espresso` + a CameraX fake frame source. This needs
either an emulator (slow on a VM) or a connected device.

### 9. Compose UI tests are not wired

We have unit tests for the data layer but nothing for the Composables.
A `commonTest` Compose UI test using `runComposeUiTest` would let us
validate the model-chip row, threshold slider, and failure-case picker
without hitting the camera. `kotlin.compose.test` is needed; we did not
add it because the harness then needs Skia native libs at JVM test time
which isn't free.

### 10. Linux Compose-Desktop GUI is not headless-tested

The `:composeApp:run` desktop entry point pops a window. There's no
"render-and-quit" test that asserts pixels of the overlay match a
golden image. This is solvable via `org.jetbrains.compose.ui.test` but
requires running an X server.

## Backend-comparison harness

### 11. CompareCli does not yet drive the Android backend

Only the JVM backend is exercised. A future command (`./gradlew
:composeApp:installDebug && adb shell am start ... compare`) would
launch the CLI variant of the app on the device, run for N frames,
write a JSON report, and `adb pull` it back. This is the natural
follow-up for Milestone 3.

## Failure-case capture

### 12. ~~No "tag while captured" UI~~ — **DONE** (527e9ae)

The failure-case save flow now opens an `AlertDialog` with an optional
free-text note field above the failure-type buttons. The note is
persisted as `userNote` in `metadata.json` and as `user_note.txt` next
to the image.

### 13. Failure cases are never automatically uploaded

This is intentional (the GOAL.md says fully offline, no upload). When
upload is wanted, add a separate `dev/sync_failure_cases.sh` script
that the user runs explicitly — never a background uploader, never a
toggleable UI, since this would cross the "no hidden network
dependency" line.

## Documentation

### 14. ~~AGENT_GOAL.md missing~~ — **DONE**

`AGENT_GOAL.md` is now a symlink to `GOAL.md` at the repo's root, so
agents that look for the canonical agent-goal filename find it without
having to read `docs/README.md`.

### 15. No screenshot, no demo gif

A 30-second screen recording on Pixel 5 with the YOLOX-nano model
running would do more for the README than any prose. Add it after the
first physical-device validation pass.

## Low-priority

### 16. Composable kotlinOptions deprecation

Gradle warns that the `kotlinOptions` DSL is being replaced by
`compilerOptions`. The current syntax still works on AGP 8.5 + KMP 2.0
but will need migration when we move to Gradle 9.

### 17. `org.jetbrains.compose.experimental.uikit.enabled=true`

Kept in `gradle.properties` for the future iOS target. It's harmless
on Android/desktop builds but should be removed if iOS is dropped.

### 18. Two thresholds (backend floor vs UI default) — convention

After review feedback, the backend score filter and the UI slider are
two distinct fields:

- `BACKEND_FLOOR_THRESHOLD = 0.01f` (constant in `DetectorBackend.kt`):
  the floor that every real backend passes to `Yolox.postprocessDecoded`.
  This is what the backend filters at before returning detections.
- `state.scoreThreshold` (mutable, UI-controlled, persisted via
  Settings): the live filter applied by the analysis loop *after* the
  backend returns. Initialised to `ModelSpec.scoreThreshold` for
  whichever model is active.

A future change to the floor (e.g. dropping it to 0.001 so the slider
can sweep wider) should keep the two-field invariant. Don't conflate
them or you'll re-introduce the "slider can't lower the model
threshold" bug. See the SettingsTest and `OnnxBackendSmokeTest` for
the contract.

### 19. `droppedFrames` counter is paused-only — naming

`FrameTelemetry.droppedFrames` increments only while the user pause
toggle is active. CameraX `STRATEGY_KEEP_ONLY_LATEST` drops frames
internally when the analyzer can't keep up; those drops are *not*
counted because CameraX doesn't expose them as events. The HUD field
is paused-drop semantics, not analyzer-busy semantics.

Fix sketch: rename to `pausedFrames`, or wire a real analyzer-busy
counter by measuring time-between-frames vs camera frame interval.
The latter would need either a separate camera-rate-known timer or
ImageAnalysis's actual frame-interval, which is not directly
exposed.
