# 003 — Known limitations and next-agent backlog

These are deliberate non-issues for the current scaffold but real concerns
for any agent who picks up the prototype next. They are not breakages.
They are the things the next pull request should address before the app
goes anywhere near a user-facing release.

## Hot-path correctness

### 1. Overlay rotation handling

The current `DetectionOverlay` assumes the analyzed frame is displayed
in the same orientation it was analyzed. CameraX `ImageAnalysis` returns
`ImageProxy` with a `rotationDegrees` field that describes how to rotate
the buffer to match natural orientation. The activity is portrait-locked,
so on Pixel 5 the back camera generally reports 90° rotation and the
underlying buffer is `[height,width]` of the displayed preview.

`ImageProxyFrame` records `rotationDegrees`. The overlay does **not**
currently consume it. In practice this is fine for portrait-only viewing
of a centered subject, but the boxes may be visibly off if the user
looks at them carefully on a real device.

Fix sketch: have `DetectionOverlay` accept a `rotationDegrees: Int`
parameter and apply the inverse rotation to the canvas before drawing,
matching what `PreviewView`'s ScaleType does internally.

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

### 5. APK is unsigned debug-only

`assembleDebug` works. `assembleRelease` is not yet wired up because
release signing is a user-managed concern. When the user is ready to
sideload signed builds, add a signing config block + a `release {}`
build type. Do not commit signing keys.

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

### 12. No "tag while captured" UI

The user can save a case but cannot type a freeform note. Adding a
small text-input dialog before save would make captures much more
useful for downstream dataset work.

### 13. Failure cases are never automatically uploaded

This is intentional (the GOAL.md says fully offline, no upload). When
upload is wanted, add a separate `dev/sync_failure_cases.sh` script
that the user runs explicitly — never a background uploader, never a
toggleable UI, since this would cross the "no hidden network
dependency" line.

## Documentation

### 14. CLAUDE.md / AGENT_GOAL.md have not been authored

`README.md` and `docs/000-002` cover the operator path, but per
`GOAL.md` §"Repo placement" we should also drop a copy or symlink at
`tpl/shitspotter-phone-app/AGENT_GOAL.md` so future agents see the
full goal without leaving the folder. This is a one-liner; left for the
next agent so the diff stays focused.

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
