# 000 — Stack decision record (ShitSpotter phone app, v2)

Status: **accepted**
Date: 2026-05-10
Author: implementation agent (autonomous run)
Supersedes: nothing
Scope: tpl/shitspotter-phone-app/

This is the Milestone-0 deliverable required by [GOAL.md](../GOAL.md). It records
the cross-platform stack choice for the new ShitSpotter phone app, what was
rejected and why, and how the choice maps onto the Linux dev VM, the
Pixel 5 target, the desktop test harness, and the future iOS path.

---

## 1. Decision summary

**Chosen stack:** Kotlin Multiplatform (KMP) + Compose Multiplatform (CMP),
Android-first, Gradle-driven.

```text
Language          Kotlin 1.9.x / 2.x (compose-compiler-aligned)
Build             Gradle (project-local wrapper) + Android Gradle Plugin 8.x
Mobile UI         Jetpack Compose on Android (via Compose Multiplatform)
Desktop UI        Compose for Desktop (JVM target, Linux native)
Camera (Android)  CameraX (Preview + ImageAnalysis, KEEP_ONLY_LATEST)
Inference         Pluggable DetectorBackend interface
                    Android: ONNX Runtime Mobile (NNAPI / CPU EP)
                    Desktop: ONNX Runtime Java (CPU)
                    iOS: scaffolded, not built from this VM
                    Future: LiteRT (TFLite), ExecuTorch, Core ML
Inference fallback Stub/fake detector (Milestone 1) so the pipeline runs
                  before any model file is present
Min Android SDK   24 (Pixel 5 ships with Android 11/12 — well above 24)
Target Android    34 (matches installed platform; required by Play in 2026)
JDK               Eclipse Temurin 17 (already installed in toolchain)
NDK               r26d (already installed in toolchain) — only needed if we
                  go down the LiteRT path; not required for ORT-mobile AAR
```

**Repo placement:** the prototype lives in-tree at
`tpl/shitspotter-phone-app/` (a temporary exception to the `tpl/* = submodule`
convention). It is intended to split out to `Erotemic/shitspotter-phone-app`
once the prototype stabilizes; the `composeApp/` subfolder is structured so
that move is mechanical.

---

## 2. Why this stack

The user constraints from `GOAL.md` that drove the choice:

1. **Android first, Pixel 5 today, app-store eventually.** Kotlin is the
   first-class Android language; Compose is the first-class Android UI; the
   AGP/Gradle path is the only one that lets us run on Pixel 5 *and* stay on
   the eventual Play Store path without rebuilding the app shell.
2. **Linux-developable, no Windows, no macOS required for Android work.**
   The KMP toolchain (JDK + Android SDK + Gradle) builds cleanly from Linux.
   The Android SDK + NDK + platform-tools are all already installed under
   `/data/tmp/shitspotter-app-toolchain/`.
3. **Linux-native test harness.** Compose for Desktop runs as a JVM app on
   Linux. The same shared-core code (model registry, detector interface,
   detection-result types, overlay geometry, telemetry) is reused against
   still images and video files on the desktop target, so CI-style
   regression of pre/post-processing does not require an Android emulator.
4. **iOS is not blocked.** KMP officially supports sharing business logic
   between Android and iOS. The shared core compiles to an iOS framework
   when a macOS build host is added later. Compose Multiplatform's iOS
   support is stable enough for the app shell; the camera + inference
   hot path will use a thin `iosMain` adapter binding to AVFoundation +
   ONNX-Runtime/CoreML EP or Core ML directly.
5. **Pluggable inference.** The `DetectorBackend` interface lives in
   `commonMain`. Each platform target supplies its own backend
   implementation, so we can switch from ONNX Runtime → LiteRT → ExecuTorch
   → Core ML per-platform without touching shared code.
6. **Hot-path discipline.** CameraX `ImageAnalysis` with
   `STRATEGY_KEEP_ONLY_LATEST` matches the `GOAL.md` hot-path spec exactly:
   drop frames when inference is slow, never queue unbounded frames, always
   close every `ImageProxy`. Frame buffers stay native on Android; only a
   compact `DetectionResult` crosses into shared code.

---

## 3. What was rejected and why

### 3.1 .NET MAUI (current MAUI app at `tpl/poopdetector/`)

Rejected. This is the explicit user intent. Reasons recorded for posterity:

- Windows toolchain dependency in practice (MSBuild, Visual Studio).
- Microsoft's mobile-MAUI camera + ML stack is not Android-first; live
  throughput in the existing app is ~3 FPS, well below the 10 FPS desired
  rate, and the JPEG-snapshot architecture is not the right hot path.
- Limited interop with the current Android-native ML runtimes (NNAPI,
  GPU delegate) compared to Kotlin-native paths.

### 3.2 Flutter

Rejected as the *primary* stack, kept as the named fallback. Why not first:

- The official `camera` Flutter plugin does not give us native Linux camera
  capture, so the desktop-test-harness story is no better than CMP.
- Real-time camera ML on Flutter forces a native plugin for the entire
  hot path anyway; the Dart layer becomes a UI shell, not a savings.
- The Dart/native bridge for image streams is an extra hop the
  Kotlin/JNI/CameraX path does not have.
- Flutter is still a strong fallback if KMP+CMP turns out to break in
  some specific way; this decision is reversible inside the same repo
  without changing the GOAL.

### 3.3 Native Android Kotlin only (no shared module)

Rejected as *primary* but kept as a graceful-degradation path. Why not:

- It works for Android, but forces a separate Swift/SwiftUI app for iOS
  and a separate Linux harness for desktop testing — three codebases for
  the same model registry, telemetry schema, and overlay geometry.
- KMP gives us the same Android-native ergonomics for the hot path *plus*
  a shared core. We only "fall back" to native-Android-only if the KMP
  build pipeline becomes a maintenance burden, and even then the
  `androidMain` source set is already structured the way a stand-alone
  Android app would be.

### 3.4 Qt/QML + C++

Rejected. Mature on Linux desktop, but mobile camera + ML runtime
integration on Android is more complex than CameraX, the licensing /
deployment story is heavier, and Kotlin is more idiomatic for Android
ML. Considered only if we hit a hard blocker on JVM-on-Android perf,
which is not a concern for camera-frame analysis (Java/Kotlin handles
ImageProxy buffers natively).

### 3.5 Rust + Slint

Rejected as the *app shell*. Rust is interesting as an eventual shared
inference/geometry core that compiles to both Android (via JNI) and iOS
(via UniFFI), but Slint's mobile camera + ML runtime story is too thin
in 2026 to bet the whole shell on. If we later want a Rust core, it
slots in under `DetectorBackend` and below — orthogonal to this choice.

### 3.6 React Native / Expo

Rejected. Linux native desktop is not first-class, the JS bridge is a
real cost on hot paths, and we would still need a native module for
the entire camera + inference path. No ergonomic win over KMP+CMP.

---

## 4. Build paths

### 4.1 Linux dev VM (this VM)

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
cd tpl/shitspotter-phone-app
./gradlew :composeApp:assembleDebug          # Android debug APK
./gradlew :composeApp:installDebug           # Android install (requires adb device)
./gradlew :composeApp:run                    # Compose Desktop on Linux
./gradlew :composeApp:test                   # JVM unit tests for shared core
```

Gradle is *not* installed globally; the project ships a Gradle wrapper. The
wrapper jar is committed (conventional and required for reproducibility),
the wrapper distribution is downloaded into `~/.gradle/wrapper/dists/` on
first run.

### 4.2 Pixel 5 (physical, user-side)

The VM has no USB passthrough. After the agent produces an APK at
`composeApp/build/outputs/apk/debug/composeApp-debug.apk`, the user runs
the install from their workstation:

```bash
# user's host workstation
adb devices                                  # confirm Pixel 5 is connected
adb install -r /path/to/composeApp-debug.apk
adb logcat -s "ShitSpotter:V"                # filter app logs
```

### 4.3 Linux desktop test target

Compose for Desktop produces a runnable JVM app:

```bash
./gradlew :composeApp:run -PdesktopHarnessImage=/path/to/test.jpg
./gradlew :composeApp:run -PdesktopHarnessVideo=/path/to/test.mp4
```

The desktop target loads still images and video frames, runs the same
preprocessing/postprocessing as Android, draws the same overlay primitives,
and prints the same telemetry schema. It does *not* require a working
webcam in this milestone.

### 4.4 iOS (future, not from this VM)

`src/iosMain/` is scaffolded so a macOS host can later run:

```bash
./gradlew :composeApp:linkDebugFrameworkIosArm64
open iosApp/iosApp.xcodeproj    # human-driven Xcode build/sign
```

iOS is **unvalidated** until a macOS build host is available. This is
called out in the report.

---

## 5. Inference runtime abstraction

`DetectorBackend` (in `commonMain`) is the only interface the shared UI /
state machine depends on. Each platform supplies an implementation in its
source set:

| Source set    | First backend                  | Future backends |
|---------------|--------------------------------|-----------------|
| `androidMain` | `OnnxRuntimeAndroidBackend` via `com.microsoft.onnxruntime:onnxruntime-android` AAR with NNAPI + CPU EP | LiteRT (TFLite) + GPU delegate, ExecuTorch |
| `desktopMain` | `OnnxRuntimeJvmBackend` via `com.microsoft.onnxruntime:onnxruntime` JAR (CPU only) | optional CUDA EP |
| `iosMain`     | scaffold; will use ONNX Runtime + CoreML EP, or native Core ML | LiteRT Core ML delegate |

The `ModelSpec` carried with each model file tells the backend everything it
needs (input size, layout, dtype, color order, normalization, postprocess
type) so the same backend can serve multiple poop-detector variants
without code changes.

---

## 6. First model artifact

Per `GOAL.md` §"Model artifact policy" and the README:

```text
tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx
```

The app references this model from outside its own folder (`tpl/poop_models/`
is a separate submodule). It is **not** copied into the in-tree app folder,
and `*.onnx` is in `.gitignore` as a defense-in-depth.

For Milestone 1, the active backend is the `StubDetectorBackend` which
returns a deterministic fake box, so the pipeline can be exercised without
the model file present. Milestone 2 swaps in `OnnxRuntimeAndroidBackend`
and the YOLOX preprocessing/postprocessing.

---

## 7. Performance instrumentation

`commonMain` defines `FrameTelemetry`:

```text
device_model           // androidMain / desktopMain / iosMain
os_version
app_commit             // injected at build time via BuildConfig
model_id, model_hash
runtime_backend        // "onnxruntime-android-1.x" etc.
delegate               // "NNAPI" / "CPU" / null
input_size             // 640x640
capture_ms             // wall-clock for frame acquisition
preprocess_ms
inference_ms
postprocess_ms
overlay_ms
fps_recent             // 1s sliding-window
detection_count
dropped_frames         // ImageAnalysis backpressure counter
```

The HUD displays `FPS`, `inference_ms`, `delegate`, and `model_id`. The
`Failure-case` capture writes the full struct as JSON next to the captured
image.

---

## 8. Risks / open items

- **CMP-Linux + Compose Compiler version drift.** Compose-Multiplatform's
  Linux desktop target is supported but moves quickly. The Gradle build
  pins compose-multiplatform and AGP versions explicitly to avoid
  silent breakage; mismatches will fail at build time, not runtime.
- **First Gradle build is large** — multi-GB of dependencies populate
  `~/.gradle/caches/`. This is the main reason the autonomous run can
  budget-bust; we attempt the build once, capture the failure mode if any,
  and document the exact next command for the user.
- **NNAPI quality on Pixel 5.** Snapdragon 765G + Adreno 620 — NNAPI may
  not accelerate every YOLOX op cleanly. The backend records the actual
  delegate that ran; the app does not silently lie about acceleration.
- **iOS path unvalidated.** Source sets and build wiring exist; no
  macOS host is available to verify they link.
- **Model conversion is out of scope.** This app does not convert models.
  It loads what is given, records the input/output schema in `ModelSpec`,
  and reports failures rather than silently mis-handling shapes.

---

## 9. What this decision does NOT do

- It does not pick the best detector model. That is a separate workstream.
- It does not pick the production app-store flow. That is a later-milestone
  decision that builds on this foundation.
- It does not commit the team to ONNX Runtime forever — only as the most
  practical first backend given existing model assets.
- It does not commit to KMP+CMP forever — Flutter and native-Kotlin remain
  graceful fallbacks if a hard blocker appears later.

---

## 10. Next steps after this document

Per `GOAL.md` §"Operating mode for the first implementation pass", the agent
proceeds without stopping at Milestone 0:

1. Scaffold `composeApp/` with `commonMain`, `androidMain`, `desktopMain`,
   and `iosMain` source sets.
2. Implement `DetectorBackend`, `ModelSpec`, `Detection`, `FrameTelemetry`,
   `FailureCase`, and `StubDetectorBackend` in `commonMain`.
3. Implement Android `MainActivity` + CameraX `ImageAnalysis` pipeline +
   overlay Composable + FPS HUD + failure-capture button.
4. Implement Desktop main + still-image harness running the same shared
   pipeline.
5. Run `./gradlew :composeApp:assembleDebug` and report the outcome.
6. Wire the ONNX Runtime backend (Milestone 2) so YOLOX-nano poop ONNX
   loads on both Android and Desktop.
7. Write `dev/journals/<date>_phone_app_milestone_0_to_2.md` summarizing
   the run.

---

This document is the source of truth for the stack choice. Future agents
should update it (with a new ADR file in `docs/`, not by overwriting this
one) if a hard blocker forces a switch.
