# Journal — 2026-05-10 — Phone-app v2 KMP+Compose scaffold

Author: implementation agent (autonomous run)
Repo state: branch `main`
Scope: `tpl/shitspotter-phone-app/`

## What this run produced

A working Kotlin Multiplatform + Compose Multiplatform skeleton for the
ShitSpotter v2 phone app. End state at the time of this journal:

- **Milestone 0 (decision record)** — done.
  See [`tpl/shitspotter-phone-app/docs/000_stack_decision.md`](../../tpl/shitspotter-phone-app/docs/000_stack_decision.md).
- **Milestone 1 (skeleton app)** — done.
  - shared core in `commonMain`: `BoundingBox`, `Detection`, `Nms`, `LetterboxParams`,
    `ModelSpec` + `ModelRegistry`, `Normalization`, `ResizePolicy`,
    `PostprocessType`, `FrameTelemetry`, `FpsCounter`,
    `LatencyAccumulator`, `FailureCaseMetadata`, `FailureType`,
    `DetectorBackend` + `StubDetectorBackend`, `Yolox`,
    `Preprocessing`, `AppState`, `AppLogger`, `AppRootTheme`,
    `AppScreen`, `DetectionOverlay`, `TelemetryHud`, `BuildInfo`.
  - Android target wired with CameraX `ImageAnalysis`
    (`STRATEGY_KEEP_ONLY_LATEST`, RGBA8888 output, `ResolutionStrategy`
    640x480 closest-higher-then-lower), permission flow, on-device
    failure-case store at `<external-files-dir>/failure_cases/`.
  - Desktop target wired with Compose for Desktop, still-image harness
    that pumps the same shared pipeline.
  - iOS target scaffolded (placeholders in `iosMain` so the source set
    compiles when a macOS host is added).
  - `commonTest` unit tests for IoU, NMS, letterbox round-trip, YOLOX
    decoded postprocess filtering, NCHW vs NHWC layout, RGB vs BGR
    swap. **All pass on `:composeApp:desktopTest`.**
  - **Android debug APK builds.** `:composeApp:assembleDebug` produced
    `composeApp/build/outputs/apk/debug/composeApp-debug.apk` (~82 MB).
- **Milestone 2 (real model backend)** — wired but not yet validated end-to-end.
  - `OnnxRuntimeAndroidBackend` (uses NNAPI EP w/ FP16, falls back to CPU).
  - `OnnxRuntimeJvmBackend` for desktop CPU testing.
  - `AndroidModelLoader` that resolves a `ModelSpec` to an absolute file:
    external files dir → cache → APK assets (copied once).
  - Desktop `Main` accepts `--model=<path>` and `--model-id=<id>`;
    falls back to stub when missing.
  - `MainActivity.chooseBackend(...)` tries to find the
    `yolox_nano_poop_cropped_only_best.onnx` artifact; falls back to
    `StubDetectorBackend()` when missing (which is the case in the
    plain APK because we deliberately don't commit weights).

The build succeeds on the Linux VM. The APK has not been installed on
a Pixel 5 (no USB passthrough from this VM).

## Stack choice (1-paragraph summary; full rationale in `docs/000_stack_decision.md`)

KMP + Compose Multiplatform, Android-first. Kotlin is first-class on
Android, CameraX `ImageAnalysis` matches the GOAL hot-path spec exactly,
the JVM target gives us a Linux-native still-image regression harness
that runs the same shared code as the phone, and `iosMain` keeps the iOS
path open for whenever a macOS host appears. Flutter and native-Kotlin
remain named fallbacks; nothing about this scaffold blocks switching.

## What still needs a human / a phone

1. **Pixel 5 sideload**: install the APK from a workstation that has the
   phone connected:

   ```bash
   adb install -r tpl/shitspotter-phone-app/composeApp/build/outputs/apk/debug/composeApp-debug.apk
   adb logcat -s "ShitSpotter.AnalysisLoop:V" "ShitSpotter.Failure:V"
   ```

2. **Drop the model in**: choose one of:

   ```bash
   # in the repo
   cp tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx \
      tpl/shitspotter-phone-app/composeApp/src/androidMain/assets/

   # OR push to the device
   adb push tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx \
      /sdcard/Android/data/io.kitware.shitspotter/files/models/
   ```

3. **Validate live FPS / inference latency** against the
   GOAL.md targets (1 FPS minimum, 10 FPS desired, 15-30 FPS excellent)
   and report the HUD readings.

4. **Pull failure cases** back to the workstation:

   ```bash
   adb pull /sdcard/Android/data/io.kitware.shitspotter/files/failure_cases/ ./
   ```

## Decisions worth recording for future agents

- **In-tree, not a submodule**: GOAL.md explicitly allows this for the
  prototype; the `composeApp/` subfolder is structured so a future split
  to `Erotemic/shitspotter-phone-app` is mechanical.
- **`*.onnx` is gitignored.** The model lives at
  `tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx` (separate
  submodule). The phone-app picks it up via `AndroidModelLoader` from
  external files dir, cache, or assets at install time. Do not copy
  the weights into the in-tree app folder.
- **`assembleDebug` produces an unsigned debug APK**. Production signing
  is out of scope until app-store packaging is in scope (per GOAL.md).
- **NNAPI is tried first, CPU is the fallback.** The actual delegate is
  recorded in `FrameTelemetry.delegate` and surfaced in the HUD, so we
  never silently lie about acceleration. Pixel 5 (Snapdragon 765G) may
  not accelerate every YOLOX op cleanly — this is expected.
- **Stub detector stays in the codebase.** It's the failure-mode for
  "model file is missing" and the fixture for UI/overlay regression
  tests. Don't remove it when adding new backends.

## What the next agent should pick up

The first three are easy follow-ups; the fourth is the fun one.

1. **Validate on Pixel 5** with a real model file. Record FPS / latency /
   delegate / dropped-frame counts and append to a Milestone-2 results
   table in `docs/`.
2. **Capture last frame as JPEG** in the failure-case path (currently
   `lastFrameJpegBytes` is a 0-byte placeholder on Android — encode the
   last analyzed `ImageProxy` as JPEG before writing the metadata).
3. **Wire model selection UI**: there's a `ModelRegistry` and a settings
   slot on `AppState.activeModelId`, but nothing in the UI lets the user
   switch backends. Should be a small Compose chip row.
4. **Optional Milestone 3**: backend comparison hook — run the same
   frame through ORT-CPU and ORT-NNAPI back-to-back, log both
   `FrameTelemetry`, write a side-by-side report. This is the foundation
   for evaluating LiteRT/ExecuTorch when those models exist.

## Build commands cheat sheet (full version: `docs/001_build_run_validate.md`)

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
cd tpl/shitspotter-phone-app
./gradlew :composeApp:desktopTest
./gradlew :composeApp:run --args="--image=/path/to/test.jpg"
./gradlew :composeApp:assembleDebug
# APK at composeApp/build/outputs/apk/debug/composeApp-debug.apk
```
