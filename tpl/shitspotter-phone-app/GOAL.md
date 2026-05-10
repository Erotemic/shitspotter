# ShitSpotter Phone App Agent Goal

This file is for an agent starting work on a replacement ShitSpotter phone app.

The current .NET MAUI app is useful as a reference, but the goal of this effort is to move away from .NET MAUI and build toward an Android-first, local-inference, battery-conscious app that can also be developed and tested natively from Linux.

Recommended location:

```text
dev/PHONE_APP_AGENT_GOAL.md
```

or, if the app prototype begins immediately:

```text
tpl/shitspotter-phone-app/AGENT_GOAL.md
```

---

## Executive summary

Build a new ShitSpotter mobile-app prototype that performs **live local poop detection** on Android first, with a Linux-native development/test path and a plausible path to iOS later.

Do not focus on finding or training the best model. That task is still open and belongs elsewhere. The app should make it easy to plug in better and faster models later.

Primary target:

```text
Android phone: Pixel 5
Current acceptable floor: 1 FPS
Desired: 10 FPS
Excellent: 15-30 FPS
Inference mode: local-only, offline
Distribution now: sideload/dev builds
Future: do not block eventual app-store release
```

The current .NET MAUI app should be treated as a reference implementation for behavior and assets, not as the preferred foundation.

---

## User intent and constraints

The user wants:

- abandon .NET MAUI;
- no dependency on Windows;
- Android first because the current phone is a Pixel 5;
- local on-device inference as the default and primary mode;
- no required internet connection;
- sideload/dev builds first;
- eventual app-store feasibility, but not app-store polish now;
- Linux-native development and testing where possible;
- a path to iOS if it does not compromise Android performance and project quality;
- the ability to use existing models now and more efficient/better-trained models later;
- a way to capture/save model failure cases to improve future training data.

The user is willing to accept tradeoffs if full Linux + Android + iOS parity conflicts with efficient Android/iOS runtime behavior.

---

## Existing repo context

The ShitSpotter repo currently has a submodule for the MAUI app:

```text
tpl/poopdetector -> https://github.com/Erotemic/poopdetector.git
branch = shitspotter
```

It also has model/framework submodules such as:

```text
tpl/poop_models
tpl/YOLOX
tpl/DEIMv2
tpl/segment-anything-2
tpl/MaskDINO
tpl/Open-GroundingDino
tpl/YOLO-v9
```

The current MAUI app appears to:

- target Android, iOS, MacCatalyst, and conditionally Windows;
- use Camera.MAUI for camera access;
- use SkiaSharp for overlays;
- use Microsoft ONNX Runtime / Microsoft ML packages;
- embed several ONNX model assets;
- run a loop that takes JPEG snapshots from the camera, converts the stream to bytes, runs model inference, and draws overlays;
- display FPS;
- support camera switching, torch, freeze/capture, accept/reject, and SAM-like mask interactions.

This app is known to sort of work, but live model throughput is laggy, around 3 FPS in current experience. Do not assume that its camera/JPEG/inference loop is the right architecture.

---

## Non-goals

Do not spend this app effort on:

- training a new detector;
- selecting the best detector architecture;
- solving the full model-efficiency problem;
- building cloud inference;
- app-store packaging;
- login/account systems;
- polished onboarding;
- feature-complete iOS release;
- reproducing the MAUI app exactly;
- requiring the full ShitSpotter dataset to test the app.

The app should expose enough runtime/model metadata and benchmarking hooks to support future model work, but model search is a separate project.

---

## Recommended stack priority

The agent should verify the current state of each framework before committing, but start from this prioritization.

### 1. Kotlin Multiplatform + Compose Multiplatform, Android-first

This is the preferred initial direction unless local investigation reveals a blocking issue.

Why:

- Android-native integration is strongest with Kotlin.
- CameraX ImageAnalysis is designed for per-frame image processing and ML inference.
- CameraX has explicit frame-dropping/backpressure strategies for slow analyzers.
- Kotlin Multiplatform is officially supported by Google for sharing business logic between Android and iOS.
- Compose Multiplatform supports Android, iOS, desktop, and web.
- Compose Multiplatform has a Linux desktop target, giving a plausible native Linux test app.
- Platform-specific source sets let Android use CameraX and Android runtimes directly while Linux uses a file/video/webcam test harness.
- The app can later add iOS-specific camera and inference bindings without compromising Android-first work.

Likely architecture:

```text
tpl/shitspotter-phone-app/
  settings.gradle.kts
  build.gradle.kts
  composeApp/
    src/commonMain/        # shared UI state, model metadata, result types
    src/androidMain/       # CameraX, Android permissions, Android inference backends
    src/desktopMain/       # Linux desktop image/video/webcam test harness
    src/iosMain/           # future iOS camera/inference bindings
```

Use shared code for:

- app state;
- model registry metadata;
- thresholds and settings;
- detection-result types;
- overlay geometry;
- failure-case metadata schema;
- preprocessing/postprocessing if practical.

Use platform code for:

- camera frame acquisition;
- hardware-accelerated inference;
- model file loading;
- permissions;
- app packaging;
- hardware profilers.

Important tradeoff:

Desktop parity should not force the Android camera/inference path into an inefficient abstraction. If necessary, desktop should test the same model metadata, preprocessing, postprocessing, and overlay logic using still images or video files, while Android uses a native CameraX pipeline.

### 2. Flutter

Flutter is the main fallback if Kotlin/Compose Multiplatform is too much friction.

Why it is attractive:

- one UI framework for Android, iOS, Linux desktop, and more;
- strong Linux desktop support;
- fast iteration and polished UI tooling;
- official camera plugin supports Android/iOS/Web image streams;
- easy to build a desktop test harness.

Why it is not first choice:

- the official camera plugin does not provide the same native Linux camera story as Android/iOS;
- high-performance frame analysis may require platform-specific native plugins anyway;
- the Dart/image-stream boundary may introduce avoidable overhead;
- inference is likely to require FFI or platform channels for efficient runtime integration;
- Android-first battery/performance may be better served by native CameraX + native runtime APIs.

Use Flutter only if the agent can keep the hot path native:

```text
camera frame -> native analyzer -> native inference -> compact result -> Flutter overlay
```

Avoid:

```text
camera frame -> Dart image copy -> Dart preprocessing -> native inference -> Dart postprocess
```

unless benchmarking proves it is acceptable.

### 3. Native Android Kotlin + separate desktop harness

This is the fallback if Android performance is clearly more important than shared app UI.

Why:

- best Android camera and runtime control;
- simplest path to efficient Pixel 5 prototype;
- direct CameraX ImageAnalysis;
- direct LiteRT / ONNX Runtime / ExecuTorch integration;
- easiest Android profiling.

Tradeoff:

- iOS becomes a later Swift/SwiftUI app or separate port;
- Linux testing becomes a separate CLI/desktop harness, not the same app UI;
- more duplicated UI later.

This is acceptable if Kotlin/Compose Multiplatform or Flutter blocks efficient live detection.

### 4. Qt/QML + C++ core

Consider only if the agent finds strong reasons.

Why:

- mature Linux/Android/iOS support;
- C++ can integrate inference runtimes directly;
- good for native desktop tooling.

Concerns:

- heavier framework;
- mobile camera integration may be more complex than Android-native CameraX;
- licensing/build/deployment complexity;
- less Android-first ergonomic than Kotlin.

### 5. Rust + Slint

Interesting but probably not first choice.

Why:

- lightweight;
- Rust-native;
- good Linux desktop story;
- Android support exists.

Concerns:

- iOS support is not as mature in stable docs;
- camera and mobile ML runtime integration are likely to require substantial platform-specific work;
- may become a framework project instead of a ShitSpotter app project.

Potential role:

- future lightweight desktop diagnostic tool;
- shared Rust inference/geometry core;
- not the first full mobile app shell unless the agent verifies mature Android+iOS camera/runtime support.

### 6. React Native / Expo

Not recommended for this prototype.

Why:

- Android/iOS app development is strong;
- Linux dev host is possible.

Concerns:

- Linux desktop native app support is not first-class in core React Native;
- JavaScript bridge and plugin complexity are not ideal for real-time camera ML;
- likely to require native modules for the entire hot path.

---

## Recommended runtime strategy

Do not choose a single permanent inference runtime yet. Build a runtime abstraction.

The initial app should support at least one existing model artifact, preferably whatever ONNX model is already easiest to run from the current MAUI app or ShitSpotter model assets. But the code should make it clear that models and runtimes are swappable.

Create an interface like:

```text
DetectorBackend
  load(model_spec)
  warmup()
  analyze_frame(frame)
  close()
```

A model spec should include:

```text
model_id
model_file
model_format: onnx | tflite | pte | coreml | other
input_width
input_height
input_layout: nchw | nhwc
input_dtype
color_order: rgb | bgr | yuv
normalization
resize_policy
letterbox_policy
output_schema
postprocess_type
threshold_defaults
class_names
model_hash
model_version
training_dataset_hint
notes
```

Preferred runtime exploration order:

### Short-term: ONNX Runtime Mobile

Reason:

- existing app already uses ONNX assets;
- low-friction path to reuse current models;
- ONNX Runtime supports mobile execution providers such as Android NNAPI and Apple CoreML.

Risk:

- some ONNX ops/models may not accelerate well;
- performance may depend heavily on model structure and execution provider support;
- Android GPU support story may differ from LiteRT.

### Medium-term: LiteRT / TensorFlow Lite

Reason:

- strong Android/iOS on-device story;
- GPU delegates can improve latency and power efficiency;
- LiteRT delegates are explicitly designed for hardware acceleration;
- newer LiteRT APIs emphasize accelerator selection, zero-copy buffers, and async execution.

Risk:

- requires model conversion;
- some models may lose accuracy or have unsupported ops;
- not the app agent's job to solve model conversion, but the app should be ready for it.

### Medium-term / PyTorch-native path: ExecuTorch

Reason:

- supports Android, iOS, and desktop;
- natural path if future ShitSpotter models are PyTorch-first;
- supports hardware backends across platforms.

Risk:

- requires exported `.pte` models and runtime integration;
- may be more moving parts than ONNX/TFLite for the first prototype.

### iOS-specific future path: Core ML

Reason:

- best Apple hardware integration;
- ONNX Runtime has a CoreML execution provider;
- LiteRT also has a Core ML delegate.

Risk:

- iOS builds and hardware testing require macOS/Xcode;
- not required for the Android-first milestone.

---

## Pixel 5 hardware note

The initial Android target is Pixel 5. It uses a Qualcomm Snapdragon 765G, Adreno 620 GPU, and 8 GB LPDDR4x RAM.

Implication:

- do not assume modern Pixel Tensor NPU behavior;
- test CPU, NNAPI, and GPU/delegate paths explicitly;
- record which delegate/backend actually ran;
- newer phones may substantially change the best backend choice, so the benchmark harness must record device model and runtime backend.

---

## Hot-path requirements

The app must avoid the current likely bottleneck pattern of repeated full JPEG capture/decode for live inference unless benchmarking proves it is acceptable.

For Android, prefer:

```text
CameraX Preview + ImageAnalysis
ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
low-resolution analyzer target suitable for the model
native YUV/RGBA conversion path
preallocated input/output buffers where possible
background inference executor
UI overlay receives compact detection results
```

The analyzer must:

- drop frames when inference is slower than camera rate;
- keep preview smooth;
- never queue unbounded frames;
- close every ImageProxy;
- record capture/preprocess/inference/postprocess/overlay timings separately;
- expose FPS and latency percentiles.

Target behavior:

```text
Preview remains responsive even if inference is slow.
Detection updates at whatever rate the backend can sustain.
The latest frame is preferred over stale queued frames.
```

---

## MVP requirements

The first prototype should provide:

1. Android app builds on Linux and installs on Pixel 5.

2. Live camera preview.

3. Live local detection overlay.

4. Offline model loading.

5. FPS and latency display.

6. Runtime telemetry:
   - device model;
   - OS version;
   - app git commit if available;
   - model id/hash;
   - runtime backend;
   - delegate/execution provider;
   - input size;
   - capture/preprocess/inference/postprocess/overlay timings;
   - detection count;
   - dropped-frame behavior.

7. Failure-case capture:
   - user can tap a button when the detector fails;
   - save image frame locally;
   - save model outputs and metadata;
   - optionally save a short note/category such as false positive, false negative, bad localization, lag, crash, uncertain;
   - do not require internet;
   - do not upload automatically.

8. Linux desktop test mode:
   - run the app or a companion desktop target natively on Linux;
   - load still images and/or video files;
   - run the same model metadata and postprocessing path where practical;
   - display overlay and timing;
   - this does not need to use a live Linux webcam in the first milestone.

9. Model registry:
   - easy to add a new model file and metadata entry;
   - model selection UI can be minimal;
   - app must not assume a single hardcoded model forever.

10. Basic docs:
    - Linux setup;
    - Android build;
    - Pixel 5 sideload command;
    - model placement;
    - known limitations;
    - how to export failure cases.

---

## Suggested milestones

### Milestone 0: decision record

Before implementing, write:

```text
tpl/shitspotter-phone-app/docs/000_stack_decision.md
```

It must answer:

- Which stack was chosen?
- Which stacks were rejected and why?
- How does it build on Linux?
- How does it run on Android?
- What is the iOS path?
- What is the Linux desktop test path?
- How are model runtimes abstracted?
- What is the first model artifact?
- What performance instrumentation exists?
- What risks remain?

Do not spend more than a focused pass on this. The goal is to choose a practical path and proceed.

### Milestone 1: skeleton app

Build:

- Android target;
- Linux desktop target or companion harness;
- simple camera preview on Android;
- placeholder detector that returns fake boxes;
- overlay rendering;
- FPS counter;
- failure-case save button that writes metadata.

Validation:

```text
./gradlew :composeApp:assembleDebug
adb install -r <debug-apk>
./gradlew :composeApp:run
```

or equivalent for the chosen stack.

### Milestone 2: first real model backend

Add:

- one existing model;
- runtime backend;
- preprocessing;
- postprocessing;
- model metadata spec;
- benchmark timings;
- overlay on real detections.

Validation:

- run on Pixel 5;
- record FPS and latency breakdown;
- run Linux still-image/video test;
- compare at least one known input against a Python/reference output if available.

### Milestone 3: runtime/backend comparison hook

Add the ability to compare:

- CPU vs accelerator delegate where available;
- ONNX vs LiteRT if both model artifacts exist;
- different input sizes if model supports them;
- Pixel 5 vs newer Android phone if available.

Do not optimize the model itself. Only make the app capable of measuring and using models.

---

## Data capture and privacy expectations

Failure-case capture is important because it helps build better training data.

The first version should save locally:

```text
failure_cases/
  YYYYMMDD_HHMMSS_<uuid>/
    image.jpg or image.png
    metadata.json
    detections.json
    optional_user_note.txt
```

Metadata should include:

```text
timestamp
device_model
os_version
app_commit
model_id
model_hash
runtime_backend
delegate
input_size
thresholds
latency_ms
fps_recent
failure_type
user_note
```

Do not upload automatically.

Future upload/sync can be added later, but the prototype must work fully offline.

---

## Performance discipline

Every performance claim must identify:

- device;
- build type: debug or release;
- runtime backend;
- delegate/execution provider;
- model id/hash;
- input resolution;
- camera frame format;
- average latency;
- p50/p90/p99 if available;
- FPS calculation method;
- whether UI preview remained smooth.

Do not report a single FPS number without context.

Minimum useful measurement table:

```text
device | build | model | runtime | delegate | input | capture | preprocess | inference | postprocess | overlay | fps
```

---

## App-store future compatibility

Do not optimize for app-store release now, but avoid obvious blockers:

- keep app id/package naming sane;
- keep privacy-sensitive permissions minimal;
- explain camera usage;
- avoid hidden network dependencies;
- avoid GPL-only code in the app unless the licensing decision is explicit;
- keep model/data licenses traceable;
- keep local data export understandable.

---

## Agent instructions

When starting work:

1. Read this file.

2. Inspect ShitSpotter repo structure and existing submodules.

3. Inspect the current MAUI app only as a reference:
   - camera behavior;
   - UI behavior;
   - model asset names;
   - current inference loop;
   - failure-case capture ideas.

4. Do not continue MAUI unless explicitly instructed.

5. Research current framework/runtime support before committing to a stack.

6. Prefer Kotlin Multiplatform + Compose Multiplatform unless it blocks Android efficiency or Linux-native testing.

7. Keep Android Pixel 5 as the first real target.

8. Keep local-only inference as a hard requirement.

9. Keep model-pluggability as a hard requirement.

10. Build the smallest prototype that proves the hot path.

11. Avoid large rewrites of unrelated ShitSpotter code.

12. Do not require the full ShitSpotter dataset.

13. Do not train or search for models.

14. Document validation honestly.

15. Update this file or add a `dev/benchmark-candidates/` entry if you discover a hard invariant future agents should preserve.

---

## Suggested initial prompt to give an implementation agent

```text
You are starting the ShitSpotter replacement phone app.

Read `dev/PHONE_APP_AGENT_GOAL.md` and inspect the ShitSpotter repo. The current
.NET MAUI app in `tpl/poopdetector` is reference-only; do not continue MAUI
unless you find a compelling reason and document it.

Goal: create an Android-first, Linux-developable prototype for live local poop
detection. It must run offline, build from Linux for Android, support Pixel 5
sideloading, and include a Linux-native desktop/image/video test path. It should
use existing model artifacts through a pluggable model/runtime abstraction.
Do not train models or search for the best model.

First write a short stack decision record comparing Kotlin/Compose
Multiplatform, Flutter, native Android Kotlin, Qt/QML, Rust+Slint, and React
Native. Start from the recommendation that Kotlin Multiplatform + Compose
Multiplatform is preferred, Flutter is the main cross-platform fallback, and
native Android Kotlin is the efficiency fallback. Verify current docs before
committing.

Then scaffold the smallest prototype:
- Android camera preview or placeholder if camera setup blocks;
- live detection overlay path, initially fake boxes if needed;
- FPS/latency display;
- model registry schema;
- local failure-case capture;
- Linux desktop still-image/video test path;
- clear build/run docs.

Prioritize the hot path:
CameraX ImageAnalysis on Android, latest-frame-only behavior, no unbounded
frame queue, no required internet, no cloud inference, and honest performance
instrumentation.

At the end, report:
- chosen stack and why;
- exact commands run;
- what works on Linux;
- what works on Pixel 5 / Android;
- what remains stubbed;
- validation results;
- next risks.
```

---

## Research notes and source links

These links were used to seed the stack recommendation. Re-check them when implementation begins.

### Kotlin / Compose Multiplatform

- Compose Multiplatform supported platforms: Android, iOS, desktop, web.
  https://kotlinlang.org/compose-multiplatform/
- Compose Multiplatform compatibility table includes Linux desktop support.
  https://kotlinlang.org/docs/multiplatform/compose-compatibility-and-versioning.html
- Google states Kotlin Multiplatform is officially supported for sharing business logic between Android and iOS.
  https://developer.android.com/kotlin/multiplatform
- Compose platform-specific APIs may require platform source-set implementations.
  https://www.jetbrains.com/help/kotlin-multiplatform-dev/compose-platform-specifics.html

### Flutter

- Flutter desktop support includes native Linux builds.
  https://docs.flutter.dev/platform-integration/desktop
- Flutter add-to-app supports Android, iOS, and web.
  https://docs.flutter.dev/add-to-app
- Flutter camera plugin supports Android, iOS, and web image streams.
  https://pub.dev/packages/camera

### Android CameraX

- CameraX ImageAnalysis is designed for per-frame image processing / CV / ML inference.
  https://developer.android.com/media/camera/camerax/analyze
- CameraX supports latest-frame-only backpressure for slow analyzers.
  https://developer.android.com/media/camera/camerax/analyze

### Inference runtimes

- ONNX Runtime NNAPI execution provider for Android.
  https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html
- ONNX Runtime CoreML execution provider for Apple platforms.
  https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- LiteRT Android quickstart and accelerator notes.
  https://ai.google.dev/edge/litert/android/quickstart
- LiteRT delegates and GPU/CoreML delegate discussion.
  https://ai.google.dev/edge/litert/performance/delegates
- ExecuTorch edge platform support.
  https://docs.pytorch.org/executorch/stable/edge-platforms-section.html

### Other frameworks

- Qt supported platforms include Android, iOS, and Linux.
  https://doc.qt.io/qt-6/supported-platforms.html
- Slint Android support.
  https://docs.slint.dev/latest/docs/slint/guide/platforms/mobile/android/
- Slint mobile docs note stable Android support and iOS roadmap concerns.
  https://docs.slint.dev/latest/docs/slint/guide/platforms/mobile/
- React Native core platform API is Android/iOS-focused; desktop/Linux are out-of-tree/community paths.
  https://reactnative.dev/docs/platform
  https://reactnative.dev/docs/next/out-of-tree-platforms

### iOS build reality

- Xcode is Apple's IDE for building apps for Apple platforms.
  https://developer.apple.com/documentation/xcode
- Xcode runs iOS apps on Simulator or real devices connected/paired to a Mac.
  https://developer.apple.com/documentation/xcode/running-your-app-in-simulator-or-on-a-device
