# ShitSpotter Phone App Agent Goal

> **Read [`README.md`](README.md) first.** It contains short-term breadcrumbs
> about the toolchain (already installed at `/data/tmp/shitspotter-app-toolchain/`),
> the first model artifact to plug in, the in-tree layout convention, and
> `.gitignore` guidance. Those are pre-flight notes that aren't repeated here.

This document is for an implementation agent starting work on a replacement ShitSpotter phone app.

The current `.NET MAUI` app is useful as a reference, but the goal of this effort is to move away from MAUI and build toward an Android-first, local-inference, battery-conscious app that can be developed and tested from Linux, with a plausible path to iOS.

Recommended location:

```text
dev/PHONE_APP_AGENT_GOAL.md
```

If the app prototype begins immediately, also copy or symlink this goal into:

```text
tpl/shitspotter-phone-app/AGENT_GOAL.md
```

---

## Executive summary

Build a new ShitSpotter mobile-app prototype that performs **live local poop detection** on Android first.

The prototype should:

- run on a Pixel 5;
- work fully offline;
- use local on-device inference;
- support live detection overlays;
- expose timing and FPS telemetry;
- save failure cases for future dataset/model improvement;
- build as much as possible from a Linux VM;
- include a Linux-native still-image/video test path;
- preserve a plausible future iOS path;
- avoid unnecessary Android/iOS code duplication;
- make it easy to plug in faster or better models later.

Do **not** focus on finding, training, or optimizing the best detection model. That task is open and belongs elsewhere. This app should provide the infrastructure needed to use current models and later swap in better ones.

Primary target:

```text
Android phone: Pixel 5
Minimum acceptable live detection rate: 1 FPS
Desired live detection rate: 10 FPS
Excellent live detection rate: 15-30 FPS
Inference mode: local-only, offline
Distribution now: sideload/dev builds
Future: do not block eventual app-store release
```

The current MAUI app should be treated as a reference implementation for behavior, assets, and lessons learned, not as the preferred foundation.

---

## Operating mode for the first implementation pass

Treat the first implementation pass as a **large autonomous session**.

Do not stop after writing a decision memo unless there is a genuine blocker. The desired behavior is:

1. inspect the repo and current app;
2. write the stack decision record;
3. choose the best practical stack;
4. install required user-local toolchain dependencies;
5. scaffold the prototype;
6. build as much as the VM can support;
7. run all feasible local validation;
8. leave exact commands for physical Android and future iOS validation.

The first deliverable is still the stack decision record, but it is not the only expected deliverable.

Do not ask for confirmation before ordinary dependency installation, project scaffolding, build execution, or local validation if the action is consistent with this document and does not require privileged system mutation.

---

## User intent and constraints

The user wants:

- to abandon `.NET MAUI`;
- no dependency on Windows;
- Android first because the current phone is a Pixel 5;
- local on-device inference as the default and primary mode;
- no required internet connection;
- sideload/dev builds first;
- eventual app-store feasibility, but not app-store polish now;
- Linux-native development and testing where possible;
- a path to iOS if it does not compromise Android performance and project quality;
- use of existing models now and more efficient/better-trained models later;
- a way to capture/save model failure cases to improve future training data;
- an elegant, extensible design;
- minimal Android/iOS code branching.

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

## Repo placement

Prototype the new app in-tree at:

```text
tpl/shitspotter-phone-app/
```

This is a temporary exception to the existing `tpl/*` submodule convention. The app may later be split into its own repository, likely:

```text
Erotemic/shitspotter-phone-app
```

Do not require that split for the first prototype.

The stack decision document should explicitly note this temporary placement and the eventual split-out intention.

The new app should not commit build products, local caches, local model binaries, or captured failure cases.

Add or update ignore rules as needed for paths such as:

```text
tpl/shitspotter-phone-app/.gradle/
tpl/shitspotter-phone-app/build/
tpl/shitspotter-phone-app/**/build/
tpl/shitspotter-phone-app/local_models/
tpl/shitspotter-phone-app/failure_cases/
```

Use equivalent ignore rules if the chosen framework creates different generated directories.

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

## Architecture principle: shared core, thin adapters

Minimize Android/iOS code branching.

Do not interpret this as “force everything through the lowest-common-denominator API.” Instead, use a shared-core architecture where almost all domain logic is common and only unavoidable device/platform integration lives in thin adapters.

The design target is:

```text
one app architecture
one model registry
one detector interface
one overlay/result model
one failure-case format
one telemetry format
thin Android/iOS/Linux adapters
```

Shared cross-platform code should own:

- model registry and model metadata;
- detector backend interface;
- app state machine;
- thresholds and settings;
- detection-result types;
- preprocessing/postprocessing semantics;
- postprocessing implementation where practical;
- coordinate transforms and overlay geometry;
- telemetry schema;
- failure-case capture schema;
- benchmark/result data structures;
- test fixtures and reference outputs.

Thin platform adapters may own:

- camera frame acquisition;
- permissions;
- file-system quirks;
- model file loading quirks;
- hardware runtime bindings;
- zero-copy or low-copy frame/tensor bridges;
- accelerator/delegate discovery and selection;
- app packaging;
- platform-specific profiling hooks.

Avoid:

```text
separate Android and iOS apps with duplicated logic
platform-specific model metadata
platform-specific overlay math
platform-specific postprocessing unless required
platform-specific failure-case formats
copying full camera frames through common code just to reduce branching
```

Efficiency still wins on the hot path. Shared code should define semantics and interfaces, but platform adapters may keep camera buffers and inference tensors native when copying would hurt latency, battery life, or hardware acceleration.

Important design rule:

```text
Minimize branching above the hot path.
Allow branching inside thin adapters where it preserves zero-copy frame handling,
native accelerator access, battery efficiency, or app-store-compatible packaging.
```

This means Android and iOS should share the same `DetectorBackend` interface and model registry, but Android may implement it with CameraX + NNAPI/GPU/ONNX/LiteRT and iOS may implement it with AVFoundation + Core ML / CoreML delegate / GPU delegate. The rest of the app should not care which platform backend produced the detections.

Desktop parity should not force the Android camera/inference path into an inefficient abstraction. If necessary, desktop should test the same model metadata, preprocessing, postprocessing, overlay logic, and telemetry using still images or video files, while Android uses a native CameraX pipeline.

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
- benchmark telemetry schema;
- detector backend interfaces;
- preprocessing/postprocessing semantics;
- preprocessing/postprocessing implementation when it does not create avoidable frame copies or block hardware acceleration.

Use platform code for:

- camera frame acquisition;
- hardware-accelerated inference;
- model file loading;
- permissions;
- app packaging;
- hardware profilers;
- zero-copy or low-copy frame/tensor bridges;
- accelerator/delegate discovery and selection.

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

## Model artifact policy

Do not commit model binaries to the new app.

Milestone 1 may use fake detections or a stub detector.

For Milestone 2, the first real model should be the YOLOX-nano poop ONNX artifact if available. Prefer referencing it from an existing model submodule or from an ignored local model directory.

Likely paths:

```text
tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx
tpl/shitspotter-phone-app/local_models/yolox_nano_poop_cropped_only_best.onnx
```

The app should use a configurable model registry so later ONNX, LiteRT/TFLite, ExecuTorch, or Core ML artifacts can be added without redesigning the app.

The app-building agent should not search for, train, or optimize the best model. That is a separate task.

---

## Pixel 5 hardware note

The initial Android target is Pixel 5. It uses a Qualcomm Snapdragon 765G, Adreno 620 GPU, and 8 GB LPDDR4x RAM.

Implication:

- do not assume modern Pixel Tensor NPU behavior;
- test CPU, NNAPI, and GPU/delegate paths explicitly;
- record which delegate/backend actually ran;
- newer phones may substantially change the best backend choice, so the benchmark harness must record device model and runtime backend.

The VM is not expected to have access to the user's Pixel 5. Physical Android validation is the user's job unless a remote ADB path is explicitly provided later.

The agent should produce:

- APK/build artifact if possible;
- exact `adb install` command;
- exact launch/logcat/benchmark commands;
- expected outputs;
- a short checklist for what the user should report back.

Do not block on lack of physical phone access.

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

1. Android app builds on Linux and can be installed on Pixel 5 by the user.

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

Before implementing code, write:

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

Do not stop at Milestone 0 unless there is a genuine blocker. This document records the decision before implementation proceeds.

### Milestone 1: skeleton app

Build:

- Android target;
- Linux desktop target or companion harness;
- simple camera preview on Android, or a clearly marked placeholder if camera setup blocks;
- placeholder detector that returns fake boxes;
- overlay rendering;
- FPS counter;
- failure-case save button that writes metadata;
- build/run documentation.

Validation:

```text
./gradlew :composeApp:assembleDebug
adb install -r <debug-apk>
./gradlew :composeApp:run
```

or equivalent for the chosen stack.

### Milestone 2: first real model backend

Suggested first model: `tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx`
(also present at `tpl/poopdetector/PoopDetector/Resources/Raw/`). Any of the
poop YOLO-v9 ONNX models in `tpl/poop_models/` are acceptable. Do **not**
copy model weights into `tpl/shitspotter-phone-app/`; reference them from
`tpl/poop_models/` and bundle into the APK at build time, or load from a
device-side path. Binary blobs do not get committed.

Add:

- one existing model;
- runtime backend;
- preprocessing;
- postprocessing;
- model metadata spec;
- benchmark timings;
- overlay on real detections.

Validation:

- build locally;
- run on Pixel 5 when the user can test;
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

## Expected autonomous stopping point

Continue until reaching the strongest feasible local result.

Preferred stopping point:

```text
- stack decision written;
- project scaffold created;
- user-local dependencies installed;
- Android debug build succeeds, if Android SDK setup is possible;
- Linux desktop/still-image harness builds or runs, if supported by chosen stack;
- fake detector or stub detector is wired to overlay/result/telemetry path;
- failure-case capture format is implemented or scaffolded;
- build/run/adb instructions are documented.
```

Acceptable fallback stopping point:

```text
- stack decision written;
- project scaffold created;
- dependency setup attempted and documented;
- blocker clearly identified;
- exact next commands provided.
```

Do not stop merely because the VM initially lacks Java, Gradle, Kotlin, Flutter, Android SDK, or similar tools. Install user-local tooling and proceed as far as practical.

---

## Dependency and VM environment policy

The implementation agent is allowed and encouraged to download, install, and cache dependencies needed to evaluate frameworks, build prototypes, run tests, and benchmark the app.

Installs should be user-local and documented. Do not use `sudo apt` or mutate the VM globally unless explicitly instructed later.

Allowed install locations include:

```text
$HOME/.sdkman/
$HOME/.local/
$HOME/.local/bin/
$HOME/.local/opt/
$HOME/Android/Sdk/
$HOME/.gradle/
$HOME/.konan/
$HOME/.cargo/
$HOME/.pub-cache/
/data/tmp/                       (large spinning disk, ~2 TiB free)
project-local Gradle wrapper
```

### Filesystem note: `/data/tmp/` vs the repo / `$HOME`

The pre-installed toolchain currently lives at
`/data/tmp/shitspotter-app-toolchain/` because that is the VM's large spinning
disk. **Do not move it.** For new work, prefer this priority order:

1. **Default — write under the repo or `$HOME`.** The shitspotter repo
   directory and `$HOME` are on regular local storage and are the right
   home for build outputs, scaffolds, scratch files, and small caches. Keep
   work here unless you have a specific reason not to.
2. **Use `/data/tmp/` only when you genuinely need the space** — multi-GB
   downloads, the toolchain, big experimental caches, large image/video
   corpora, etc. Document anything you put there.

`/data/tmp/` is mounted differently than the repo and can hit `EMFILE` /
"Too many open files" / EIO storms when the host side is under pressure.
**If `/data/tmp/` starts erroring this way, do not work around it with
suppressions or copying files around — fall back to the repo dir or `$HOME`,
note it in the report, and ask the user to reset the mount.** The repo
directory is not on the same problematic mount, so simple build work can
continue while the user investigates.

Allowed dependencies include, when appropriate:

- JDK 17+;
- Android SDK / command-line tools;
- Android build tools;
- Android platform packages;
- Gradle / Gradle wrapper;
- Kotlin / Compose Multiplatform dependencies;
- Flutter SDK and packages, if Flutter is chosen or evaluated;
- Rust / Cargo dependencies, if Rust tooling is chosen or evaluated;
- Qt / CMake / Ninja dependencies, if Qt is chosen or evaluated;
- ONNX Runtime / LiteRT / ExecuTorch packages;
- model runtime binaries;
- small public sample models or test assets;
- emulator, profiling, and benchmarking tools.

If a dependency is large or slow to install, note it in the final report. Do not stop solely because it is large unless it is clearly unreasonable for the VM or requires privileged installation.

Prefer reproducible setup:

```text
project-local wrapper > user-local SDK/toolchain > undocumented global install
```

The VM may not have:

- USB access to a physical phone;
- Android emulator hardware acceleration;
- camera hardware;
- webcam passthrough;
- GPU/NPU delegates;
- realistic battery or thermal behavior;
- macOS;
- Xcode;
- iOS signing credentials.

These limitations should not prevent useful progress.

Distinguish clearly between:

```text
VM validation:
  - dependency resolution
  - project generation
  - static checks
  - unit tests
  - desktop still-image/video harness
  - model metadata parsing
  - preprocessing/postprocessing tests
  - Android APK build, if SDK/tooling is available

Physical Android validation:
  - install on Pixel 5
  - live camera preview
  - live local detection overlay
  - real FPS and latency
  - battery/thermal behavior
  - runtime delegate behavior
  - failure-case capture from camera

Physical iOS validation:
  - iOS build/signing
  - camera integration
  - Core ML / delegate behavior
  - app-store-relevant platform constraints
```

When a required hardware feature is unavailable in the VM, do not stop. Instead:

1. document the limitation;
2. complete the closest useful local validation;
3. provide the exact command that should be run on the host or device;
4. keep the prototype structured so device validation can be performed later.

---

## iOS validation

Do not attempt to build or sign iOS artifacts from this VM unless macOS/Xcode tooling is explicitly available.

It is acceptable to include iOS source-set scaffolding or design notes, but mark iOS as unvalidated unless it is actually built and run on Apple tooling.

On iOS, the framework should be treated as the app shell and shared-logic host, not necessarily the inference engine. The camera and inference hot path should use native iOS mechanisms where needed:

```text
Compose Multiplatform UI / shared state
  -> iosMain native bridge
  -> AVFoundation frame acquisition
  -> Core ML / ONNX Runtime CoreML EP / LiteRT delegate / ExecuTorch backend
  -> compact detection results back to shared UI state
```

Do not route full camera frames through common Kotlin code solely to reduce branching if benchmarking shows that this hurts latency, battery, or accelerator access.

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

10. Minimize Android/iOS branching above the detector backend interface.

11. Build the smallest prototype that proves the hot path.

12. Install required dependencies user-locally and document them.

13. Avoid large rewrites of unrelated ShitSpotter code.

14. Do not require the full ShitSpotter dataset.

15. Do not train or search for models.

16. Document validation honestly.

17. Update this file or add a `dev/benchmark-candidates/` entry if you discover a hard invariant future agents should preserve.

---

## Final report requirements

Every implementation report should explicitly state:

```text
What was built:
What ran inside the VM:
What could not run inside the VM:
What requires a physical Android device:
What requires macOS / iOS tooling:
What dependencies were downloaded or added:
Where dependencies were installed:
What validation commands were run:
What is stubbed or fake:
What artifact or APK was produced:
What command should the user run on Pixel 5:
What the next recommended step is:
```

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

Then continue autonomously as far as practical:
- install user-local dependencies;
- scaffold the prototype under `tpl/shitspotter-phone-app/`;
- build Android and Linux targets where possible;
- create a live detection overlay path, initially fake boxes if needed;
- add FPS/latency display;
- add model registry schema;
- add local failure-case capture;
- add Linux desktop still-image/video test path;
- document clear build/run/adb commands.

Prioritize the hot path:
CameraX ImageAnalysis on Android, latest-frame-only behavior, no unbounded
frame queue, no required internet, no cloud inference, and honest performance
instrumentation.

Minimize Android/iOS branching above the detector backend interface. Keep shared
model metadata, result types, overlay geometry, telemetry, and failure-case
formats common. Allow thin platform adapters for camera acquisition, permissions,
runtime bindings, and accelerator/delegate selection.

At the end, report:
- chosen stack and why;
- exact commands run;
- dependencies installed and where;
- what works on Linux;
- what works in the VM;
- what requires Pixel 5 validation;
- what requires macOS/iOS tooling;
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
