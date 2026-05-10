# 005 — Runtime architecture

This document is the "where does each piece live and how do they talk
to each other" view of the v2 phone app. It complements the strategic
[`000_stack_decision.md`](000_stack_decision.md) (which records *why*
KMP+Compose was chosen) and the operator [`001_build_run_validate.md`](001_build_run_validate.md)
(which records *how* to build and run).

## High-level data flow

```text
                +----------------------------+
                |      camera surface        |
                |  (Android CameraX preview  |
                |   or desktop still image)  |
                +-------------+--------------+
                              |
                              v
        +---------------------+---------------------+
        |   FrameSource (commonMain interface)      |
        |   width, height, rotationDegrees,         |
        |   toRgb888()                              |
        +---------------------+---------------------+
                              |
                              v
                +-------------+--------------+
                |   DetectorBackend          |
                |   (interface, commonMain)  |
                +-+-----+----------+--+------+
                  |     |          |  |
                  v     v          v  v
        StubDetector  ORT-Android  ORT-JVM  TfliteStub (placeholder)
        (commonMain)  (androidMain) (desktopMain)  (commonMain, throws)
                  |     |          |  |
                  +-----+----------+--+
                        |
                        v
        +---------------+----------------+
        |   InferenceResult              |
        |   detections, pre/inf/post ms, |
        |   backendName, delegate        |
        +---------------+----------------+
                        |
                        v
            +-----------+------------+
            | platform analysis loop |
            |   - capture timing     |
            |   - score-threshold    |
            |   - rotation mapping   |
            |   - FPS / latency calc |
            |   - JPEG snapshot      |
            +-----------+------------+
                        |
                        v
            +-----------+----------+
            |     AppState         |
            |  (mutableStateOf)    |
            |  detections, telemetry|
            +-----+----------+-----+
                  |          |
                  v          v
        +---------+---+   +--+----------+
        | Composable  |   | failure store
        | overlay+HUD |   | (Android+desktop)
        +-------------+   +-------------+
```

## Per-package responsibilities

### `core/` (commonMain)

Pure-Kotlin shared logic. No platform classes. Always compilable.

| File | Owns |
|------|------|
| `Geometry.kt` | `BoundingBox`, `Detection`, `Nms`, `LetterboxParams`, `BoundingBox.rotated`, `filterByScore` |
| `ModelSpec.kt` | `ModelSpec`, enums (`ModelFormat`, `InputLayout`, `ColorOrder`, `ResizePolicy`, `PostprocessType`), `ModelRegistry` |
| `Yolox.kt` | `postprocessDecoded`, `decodeRawStrides` (raw-strides fallback) |
| `Preprocessing.kt` | `letterboxRgb`, `stretchRgb`, `centerCropRgb`, `toFloatTensor` (RGB↔NCHW/NHWC, normalisation, BGR swap) |
| `DetectorBackend.kt` | `DetectorBackend` interface, `FrameSource`, `InferenceResult`, `StubDetectorBackend`, `nowMonoMs` (expect) |
| `Telemetry.kt` | `FrameTelemetry`, `FpsCounter`, `LatencyAccumulator` |
| `FailureCase.kt` | `FailureType`, `FailureCaseMetadata` |
| `FailureCaseStore.kt` | interface only; per-platform impls live in `androidMain` / `desktopMain` |
| `BackendComparison.kt` | `BackendRunRow`, `BackendComparisonReport`, `BackendComparison.runMeasured`, `renderTable` |
| `AppState.kt` | the single source of truth for live state shared with Composables |
| `Settings.kt` | `AppSettings`, `SettingsStore` interface, `InMemorySettingsStore`, AppState ⇄ AppSettings extensions |
| `Logging.kt` | `AppLogger`, `PrintlnLogger` |
| `Format.kt` | `Fmt.ms`, `Fmt.ms2`, `Fmt.score` (multiplatform-safe number formatters) |
| `BuildInfo.kt` | expect; actuals fill in deviceModel / osVersion / appCommit |
| `TfliteBackendStub.kt` | placeholder DetectorBackend that throws NotImplementedError |

### `ui/` (commonMain)

Compose Multiplatform surfaces. Pure rendering against `AppState`.

| File | Owns |
|------|------|
| `Overlay.kt` | `DetectionOverlay`, `OverlayScaleMode` |
| `Hud.kt` | `TelemetryHud` (uses `Fmt.ms`) |
| `AppScreen.kt` | `AppScreen`, `AppRootTheme`, `CameraSurface` interface, model chip row, threshold slider, HUD/overlay/front-camera toggle row, failure-case picker (with note field) |

### `androidMain/`

Thin platform adapter for Android. Owns CameraX, ONNX Runtime
Android, permissions, file-system quirks, and the launcher activity.

| File | Owns |
|------|------|
| `MainActivity.kt` | activity lifecycle, permission flow, backend selection, settings load/save |
| `CameraAnalysisLoop.kt` | CameraX `ImageAnalysis` + `STRATEGY_KEEP_ONLY_LATEST`, frame timing, rotation mapping, score-threshold filter, ImageProxy → JPEG, front/back rebind |
| `AndroidCameraSurface.kt` | `CameraSurface` impl that wires PreviewView + `CameraAnalysisLoop` |
| `AndroidActuals.kt` | `nowMonoMs`, `BuildInfo` actual using `BuildConfig.APP_GIT_COMMIT` |
| `AndroidModelLoader.kt` | resolves a `ModelSpec` to an absolute file via external-files-dir → cache → APK assets |
| `AndroidFailureCaseStore.kt` | writes JPEG + JSON under `<external-files-dir>/failure_cases/<ts>/` |
| `AndroidSettingsStore.kt` | SharedPreferences-backed JSON blob |
| `OnnxRuntimeAndroidBackend.kt` | NNAPI EP w/ FP16, CPU fallback, shape validation, YOLOX postprocess via shared core |

### `desktopMain/`

Thin platform adapter for the Linux test harness.

| File | Owns |
|------|------|
| `Main.kt` | application entry, `--image` / `--frames` / `compare` / `describe` dispatch, settings save on close |
| `DesktopHarness.kt` | runOnce / runLoop / runDirectoryLoop drivers that pump the same shared pipeline |
| `DesktopCameraSurface.kt` | `CameraSurface` impl that draws a still image with `OverlayScaleMode.FIT_CENTER` |
| `StillImageFrameSource.kt` | `BufferedImage` → `FrameSource` |
| `FrameDirectorySource.kt` | replay a directory of stills as a synthetic 30 FPS feed |
| `OnnxRuntimeJvmBackend.kt` | CPU-only ONNX Runtime, shape validation, shared YOLOX postprocess |
| `DesktopFailureCaseStore.kt` | filesystem-backed FailureCaseStore |
| `FileSettingsStore.kt` | `~/.shitspotter/settings.json` |
| `CompareCli.kt` | `compare` + `describe` subcommands |
| `DesktopActuals.kt` | `nowMonoMs`, `BuildInfo` actual |

### `iosMain/`

Scaffolded only. Cannot build from Linux.

| File | Owns |
|------|------|
| `IosActuals.kt` | `nowMonoMs` (UNIX epoch ms via `NSDate`), `BuildInfo` actual via `UIDevice` |

The corresponding `iosArm64()` / `iosX64()` / `iosSimulatorArm64()`
targets in `composeApp/build.gradle.kts` are gated behind a
non-Linux host check or an explicit `-Pssp.enableIosTargets=true`.

## Hot path

Per GOAL.md §"Hot-path requirements":

```text
[CameraX] ImageProxy (RGBA8888, KEEP_ONLY_LATEST)
    → ImageProxyFrame.from(proxy)             # 1 buffer copy (RGBA bytes)
    → backend.analyze(frame)                   # main inference call
        → frame.toRgb888()                     # 1 buffer copy (RGBA→RGB)
        → Preprocessing.letterboxRgb           # 1 buffer copy (letterboxed RGB)
        → Preprocessing.toFloatTensor          # 1 buffer copy (uint8→float NCHW)
        → OrtSession.run                       # native, in-place
        → Yolox.postprocessDecoded             # native arrays only, no copy
        → letterboxParams.mapBoxToSource       # cheap, per-box
    → score-threshold filter                   # cheap, per-box
    → BoundingBox.rotated                      # cheap, per-box
    → AppState.pushFrame                       # mutableStateOf write
    → DetectionOverlay redraw                  # Compose recompose
```

Total buffer copies on the hot path: 4 (RGBA in, RGBA→RGB, letterbox,
RGB→float). Each is unavoidable without zero-copy YUV→tensor through
RenderScript / Vulkan compute (see `docs/003_known_limitations.md` #4).

The `proxy.close()` call is in a `finally` block, so every ImageProxy
is released exactly once regardless of analysis outcome.

## Backpressure

CameraX's `STRATEGY_KEEP_ONLY_LATEST` drops frames internally when
the analyzer is slower than the camera. We do not see those drops as
events — only the frames that actually reach `setAnalyzer`. The
`droppedFrames` counter we expose in the HUD currently counts only
frames dropped by the user pause toggle; CameraX-level drops are
implicit in the FPS gap between camera rate and inference rate.

## Lifecycle

| Event | Component | Action |
|-------|-----------|--------|
| `Activity.onCreate` | MainActivity | settings load → backend init (with warmup) → camera surface composes → CameraX bind |
| `Activity.onPause` | MainActivity | settings save (SharedPreferences) |
| `Activity.onDestroy` | MainActivity | `backend.close()` |
| Window close | Desktop | settings save → `exitApplication()` |
| Frame received | CameraAnalysisLoop | rebind-if-camera-changed → handleFrame → push to AppState |
| `Save failure` tap | AppScreen | failure picker → JPEG snapshot → JSON write |
