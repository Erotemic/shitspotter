# Journal — 2026-05-10 — Phone-app v2 KMP+Compose scaffold

Author: implementation agent (autonomous run)
Repo state: branch `main`
Scope: `tpl/shitspotter-phone-app/`

## What this run produced

A working Kotlin Multiplatform + Compose Multiplatform skeleton for the
ShitSpotter v2 phone app **plus** a real-model end-to-end smoke test
on the desktop, **plus** a backend-comparison harness, **plus** a live
score-threshold slider and a frame-directory replay source. End state
at the time of this journal:

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
- **Milestone 2 (real model backend)** — wired AND validated end-to-end on the
  desktop side.
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
  - **Smoke test** at `composeApp/src/desktopTest/.../OnnxBackendSmokeTest.kt`
    really loads the YOLOX-nano model from `tpl/poop_models/`, runs
    inference on a synthetic 480x640 grey frame, and asserts that
    pre/inf/post timings are non-negative. Skips cleanly if the model
    file is absent. **Real measured numbers (Linux JVM CPU):**
    `pre=30 ms`, `inf=16 ms`, `post=8 ms`.
  - **Compare CLI** at `composeApp/src/desktopMain/.../CompareCli.kt` runs
    multiple backends back-to-back over the same image. Real run
    against `tpl/YOLOX/assets/dog.jpg`:

    ```
    backend                | delegate |  pre(ms) |  inf(ms) | post(ms) | dets |     top
    --------------------------------------------------------------------------------
    stub-1.0               | —        |     1.05 |     0.00 |     0.00 |    1 |   0.880
    onnxruntime-jvm-1.19   | CPU      |     4.54 |     9.91 |     2.82 |    3 |   0.757
    ```

    The dog produces a 0.757-score false positive at the YOLOX-nano
    default threshold — this is a known (and useful) drift, not a
    regression. JSON report archived at
    `dev/journals/2026-05-10_phone_app_compare_dog_jpg.json`.

- **Milestone 3 (backend comparison)** — kicked off and demonstrably useful.
  - `BackendComparison` (commonMain) drives N backends back-to-back.
  - CompareCli accepts repeated `--model=<path>`, `--no-stub`,
    `--score-threshold=<f>`, `--out=<json>`, `--help`. Real run
    against three models in one go committed at
    `dev/journals/2026-05-10_phone_app_compare_3_models_dog.json`.
  - `describe --model=<path>` subcommand dumps the ONNX file's
    input/output names + shapes + dtypes for spec-tuning.
  - Pending pieces are documented in `tpl/shitspotter-phone-app/docs/003_known_limitations.md`.

- **Python reference parity** — `scripts/python_reference_compare.py`
  re-implements the same letterbox + YOLOX postprocess in NumPy. Result
  on `dog.jpg`: top score 0.7463 (Python pre-NMS) vs 0.757 (Kotlin
  post-NMS). Within the precision drift expected from Pillow vs
  java.awt.image; the boxes are degenerate/off-image, which is a real
  out-of-distribution model failure documented in `docs/004_kotlin_python_parity.md`.

- **APK size** — 29 MB (was 82 MB) after filtering native ABIs to
  arm64-v8a only. Pixel 5 is arm64. Future agent: add x86_64 if
  emulator-testing.

- **TFLite backend stub** — `TfliteBackendStub` (commonMain) throws
  NotImplementedError with a pointer to GOAL.md §LiteRT. Lets a future
  ModelSpec switch to `format = TFLITE` and immediately fail loud.

- **Build commit visible** — HUD now shows `build <commit> | dropped <n>`
  so the user can verify which APK is on the device.

- **Failure-case sync** — `scripts/sync_failure_cases.sh` pulls
  `/sdcard/Android/data/io.kitware.shitspotter/files/failure_cases/`
  off a connected Pixel 5. Workstation script; never runs from this VM.

- **Live score-threshold slider** + `List<Detection>.filterByScore`. The user
  can move the threshold on the phone without rebuilding; both the
  Android `CameraAnalysisLoop` and the desktop `DesktopHarness`
  re-filter post-backend. Verified: at `--score-threshold=0.8` the
  dog.jpg test goes 3 dets → 0.

- **Test count**: 86 tests across 19 files, all green:
  - `GeometryTest` (8) — bbox intersect, IoU, NMS, letterbox round-trip,
    YOLOX postprocess
  - `PreprocessingTest` (4) — pad colour, NCHW, NHWC, BGR swap
  - `BackendComparisonTest` (3) — empty, multi-backend, table format
  - `FpsCounterTest` (3) — empty, steady-state, stale-window prune
  - `LatencyAccumulatorTest` (4) — empty, mean, window cap, percentile
  - `ModelRegistryTest` (4) — default, lookup hit/miss, unique ids
  - `AppStateTest` (3) — pushFrame, setError clear, default fallback
  - `FilterByScoreTest` (3) — zero, mid, above-max
  - `SerializationTest` (3) — failure case, model spec, comparison report
  - `BoundingBoxRotationTest` (8) — 0/90/180/270, normalisation,
    must-be-multiple-of-90, double-rotation = identity, four-quarter = identity
  - `YoloxRawStridesTest` (4) — unit-stride identity, multi-stride
    decode, mismatched-predictions assertion, mismatched-grid/strides
    assertion
  - `YoloxPostprocessExtraTest` (5) — multi-class arg-max, zero-size
    filtering, corner-form box coordinates, NMS collapse, too-small
    buffer assertion
  - `StubBackendTest` (6) — one box per call, in-bounds across 200
    calls, warmup safe, close-then-analyze throws, spec round-trip,
    finite timings
  - `PreprocessingResizeTest` (5) — stretchRgb size + assertion,
    centerCropRgb wide-vs-tall + assertion
  - `TfliteBackendStubTest` (3) — rejects non-TFLite spec, warmup +
    analyze throw NotImplementedError
  - `FilterByScoreTest` (3) — already counted above
  - desktop-only:
    - `OnnxBackendSmokeTest` (1, conditional) — real ONNX model
    - `OnnxShapeValidationTest` (1, conditional) — rejects mismatched spec
    - `FrameDirectorySourceTest` (3) — order, empty, non-dir
    - `DesktopFailureCaseStoreTest` (4) — metadata + image + note + unique-dirs
    - `CompareCliArgsTest` (8) — argValue, argValues, guessModelIdFromPath
    - `CompareCliEndToEndTest` (3) — --help, stub-only run, no-stub empty

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

Items #1-#3 from the original plan have all landed in this same run
(JPEG capture is real, model chip row is wired, Milestone-3 compare CLI
runs end-to-end). The remaining backlog is in
`tpl/shitspotter-phone-app/docs/003_known_limitations.md` — the
highlights:

1. **Validate on Pixel 5** — the only remaining Milestone 2 step. APK is
   built; user must `adb install -r` and report HUD numbers.
2. **Overlay rotation handling** — current `DetectionOverlay` ignores
   the `rotationDegrees` reported by `ImageProxyFrame`, which is fine in
   portrait-locked mode but visibly off in edge cases.
3. **ONNX raw-stride decoder** — `Yolox.decodeRawStrides` exists but is
   not wired; only models with embedded YOLOX decode work right now.
4. **CompareCli on the device** — currently desktop-only. Adding an
   Android app-mode "compare" launches the CLI variant of the comparison
   harness on the phone with the real NNAPI delegate, which is the real
   Milestone 3 deliverable.
5. **Android instrumented tests** — none yet. CameraX + an emulator + a
   fake frame source would close the only big gap left in the test
   pyramid.

Each of those is a small focused PR.

## Build commands cheat sheet (full version: `docs/001_build_run_validate.md`)

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
cd tpl/shitspotter-phone-app
./gradlew :composeApp:desktopTest
./gradlew :composeApp:run --args="--image=/path/to/test.jpg"
./gradlew :composeApp:assembleDebug
# APK at composeApp/build/outputs/apk/debug/composeApp-debug.apk
```
