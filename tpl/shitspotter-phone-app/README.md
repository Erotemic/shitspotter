# shitspotter-phone-app — agent quick-start

> An autonomous agent picking up this folder should read this **before** reading
> `GOAL.md`. The points below are short-term breadcrumbs (toolchain location,
> first model, in-tree convention) that aren't part of the long-form goal.

This directory is the new **Android-first, locally-developed** replacement for
the .NET MAUI app at `tpl/poopdetector/`. The full goal, milestones, and
constraints are in [GOAL.md](GOAL.md). Read it next.

This directory lives **in-tree** in the shitspotter repo. It is not a git
submodule (unlike every other `tpl/*` entry). Build artifacts (`build/`,
`.gradle/`, `local.properties`, etc.) and any model files belong in
`.gitignore`, not in commits.

---

## 0. Toolchain — already installed

Most of the heavy toolchain is already on the VM. **Source the env file before
running any build, gradle, adb, flutter, or cargo command:**

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
# or via the convenience symlink:
source ~/.local/share/shitspotter-app-toolchain/env.sh
```

What's installed:

| Component | Version | Path |
|---|---|---|
| Eclipse Temurin JDK | 17.0.12+7 | `$JAVA_HOME` |
| Android SDK cmdline-tools | 12.0 | `$ANDROID_HOME/cmdline-tools/latest` |
| Android platform-tools (incl. `adb`) | 37.0.0 | `$ANDROID_HOME/platform-tools` |
| Android `platforms;android-34` | — | `$ANDROID_HOME/platforms/android-34` |
| Android `build-tools;34.0.0` | — | `$ANDROID_HOME/build-tools/34.0.0` |
| Android NDK | 26.3.11579264 | `$ANDROID_NDK_HOME` |
| Flutter SDK (stable) | 3.24.5 | `$FLUTTER_HOME` |
| Rust + Cargo (stable, with Android targets) | rustup-managed | `$CARGO_HOME` |

`Gradle` is intentionally not installed globally — the per-project gradle
wrapper handles that.

If anything looks missing, re-run the installer (idempotent):

```bash
bash ./install_toolchain.sh --all          # everything
bash ./install_toolchain.sh                # just JDK + Android SDK
bash ./install_toolchain.sh --check        # verify
bash ./install_toolchain.sh --print-env    # show env.sh contents
```

The toolchain root defaults to `/data/tmp/shitspotter-app-toolchain` (the VM
has 2+ TiB free there; only ~18 GiB on `/`). Override with
`TOOLCHAIN_ROOT=/some/path bash ./install_toolchain.sh`.

> **Filesystem note.** `/data/tmp/` is the large spinning disk and can hit
> `EMFILE`/EIO storms when the host is under pressure. For *new* work, prefer
> writing under the repo directory or `$HOME` — those are on regular local
> storage and are not on the problematic mount. Use `/data/tmp/` only when
> you actually need the space (the toolchain qualifies; small build outputs
> do not). If `/data/tmp/` starts erroring, fall back to the repo dir, note
> it in the report, and ask the user to reset the mount — don't paper over
> the errors. See **GOAL.md → Dependency and VM environment policy** for
> the full rule.

---

## 1. First model artifact

For Milestone 2 (first real model backend), start with one of these existing
ONNX files — both are the same YOLO-style poop detector and are known to load
in ONNX Runtime Mobile:

```text
tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx
tpl/poopdetector/PoopDetector/Resources/Raw/yolox_nano_poop_cropped_only_best.onnx
```

Any of the YOLO-v9 poop models in `tpl/poop_models/` are also fine if you'd
rather start there.

**Do not commit `.onnx`/`.tflite`/`.pte`/weights into this folder.** The
"no binary data in git" rule applies. Reference the model from `tpl/poop_models/`
(read at build time or copied into the APK at `assembleDebug`), or have the
runtime load it from a path on the device.

The DVC roots `/data/joncrall/dvc-repos/shitspotter_dvc/` and `…_expt_dvc/`
are mounted **read-only** for the agent — read freely, never write.

---

## 2. In-tree layout convention

Recommended scaffold (the GOAL leaves the exact framework choice open, but
this is the suggested layout if KMP+Compose is chosen):

```text
tpl/shitspotter-phone-app/
  README.md                    # this file
  GOAL.md                      # full goal + milestones
  install_toolchain.sh         # toolchain installer (already run)
  build.gradle.kts             # root Gradle build
  settings.gradle.kts
  gradle/                      # version catalog + wrapper (jar IS committed)
  gradlew, gradlew.bat
  docs/
    000_stack_decision.md      # Milestone 0 — picked KMP + Compose Multiplatform
    001_build_run_validate.md  # operator checklist (build / sideload / pull)
    002_benchmarks_template.md # benchmark report schema + Pixel 5 stub
  composeApp/
    build.gradle.kts           # Android + JVM target wiring
    src/commonMain/            # shared UI state, model registry, result types,
                               # YOLOX postprocess, NMS, telemetry,
                               # backend-comparison harness
    src/commonTest/            # 30 JUnit-style tests (geometry, NMS,
                               # letterbox, YOLOX, preprocess, FPS, model
                               # registry, AppState, BackendComparison)
    src/androidMain/           # CameraX KEEP_ONLY_LATEST analysis loop,
                               # ONNX Runtime Android backend with NNAPI EP,
                               # AndroidModelLoader (external/cache/assets),
                               # AndroidFailureCaseStore, MainActivity
    src/desktopMain/           # Compose for Desktop main, ONNX Runtime JVM
                               # backend, still-image harness, CompareCli
    src/desktopTest/           # ONNX smoke test (loads real model when
                               # available, skips cleanly otherwise)
    src/iosMain/               # actuals only — needs a macOS host to build
  failure_cases/               # gitignored — runtime-captured user reports
  .gitignore                   # build/, .gradle/, local.properties, *.onnx, ...
```

`.gitignore` minimum (commit the gradle wrapper jar; it's conventional and
required for reproducibility):

```gitignore
# Gradle / Android build outputs
build/
.gradle/
local.properties
*.iml
.idea/
captures/

# Flutter (if used)
.dart_tool/
.flutter-plugins
.flutter-plugins-dependencies

# Rust (if used)
target/

# Model assets — never commit binaries
*.onnx
*.tflite
*.pte
*.pt
*.ckpt
*.safetensors

# Captured failure cases — local only
failure_cases/
```

---

## 3. Pixel 5 validation

The VM has no USB passthrough to the user's Pixel 5. The agent can build
APKs and emulator-test, but on-device validation requires the user to run
the install command on their workstation. Produce the APK and report the
exact `adb install -r <apk>` line; do not block on Pixel 5 validation.

---

## 4. Multi-agent coordination

This repo follows a `CHANGELOG.md` + journal convention so other agents can
pick up the thread. After any non-trivial milestone:

- update the repo-root `CHANGELOG.md` (or this folder's, if it grows one);
- drop a journal entry under `dev/journals/` describing what worked, what
  didn't, and what the next agent should pick up.

---

## 5. Sanity-check before starting work

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
java -version
adb --version
sdkmanager --version
flutter --version       # only if you'll use Flutter
cargo --version         # only if you'll use Rust
df -h /data/tmp /home   # confirm there's room before a big build
```

---

## 6. Quick-start (after the scaffold has been built)

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
cd tpl/shitspotter-phone-app

# One-shot end-to-end validation (toolchain + tests + compare CLI +
# python parity + APK build); use this as the first sanity check after
# every non-trivial change:
scripts/run_all_desktop_validation.sh

# All shared-core unit tests (~5 s after first build):
./gradlew :composeApp:desktopTest

# Compose for Desktop GUI (still-image harness, stub detector):
./gradlew :composeApp:run --args="--image=/path/to/test.jpg"

# Backend-comparison CLI (stub vs ONNX Runtime CPU on the same image):
./gradlew :composeApp:run --args="compare \
   --image=/path/to/test.jpg \
   --model=../poop_models/yolox_nano_poop_cropped_only_best.onnx \
   --runs=10 --warmup=2 --out=/tmp/compare.json"

# Android debug APK (~82 MB; ONNX Runtime native libs included):
./gradlew :composeApp:assembleDebug
# → composeApp/build/outputs/apk/debug/composeApp-debug.apk

# On a workstation with a Pixel 5 plugged in (NOT this VM):
adb install -r composeApp/build/outputs/apk/debug/composeApp-debug.apk
adb logcat -s "ShitSpotter.AnalysisLoop:V" "ShitSpotter.Failure:V"
```

Full operator checklist in [`docs/001_build_run_validate.md`](docs/001_build_run_validate.md).
Benchmark report schema in [`docs/002_benchmarks_template.md`](docs/002_benchmarks_template.md).

---

## 7. UI layout (Pixel 5, portrait)

The app is single-screen. Live camera preview fills the background.
Detection boxes are drawn on top in red. The HUD sits in the top-left,
the model/threshold controls follow it, and the failure-case + pause
buttons sit at the bottom.

```text
┌─────────────────────────────────────────────────┐
│ ┌────────────────────────────┐                  │
│ │ FPS 12.3   dets 1          │   ┌──────────┐   │
│ │ inf 65.2 ms  pre 8.4 post 2│   │          │   │
│ │ onnxruntime-android | NNAPI│   │  RED     │   │
│ │ model yolox-nano-poop-…    │   │  BOX     │   │
│ │ build 1bedbbe | dropped 0  │   │          │   │
│ └────────────────────────────┘   └──────────┘   │
│ [● YOLOX-nano poop] [Custom v5] [Custom v2]     │
│ ┌────────────────────────────┐                  │
│ │ score ≥ 25%                │                  │
│ │ ──────●────────────────    │                  │
│ └────────────────────────────┘                  │
│                                                 │
│              ( camera preview )                 │
│                                                 │
│                                                 │
│  ┌─────────────────────────┐ ┌─────────────┐   │
│  │ Save failure (3)        │ │ Pause       │   │
│  └─────────────────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────┘
```

Tapping `Save failure` brings up a column of failure-type buttons
(`FALSE_POSITIVE` / `FALSE_NEGATIVE` / `BAD_LOCALIZATION` / `LAG` /
`CRASH` / `UNCERTAIN` / `OTHER`); pick one and the current frame's
JPEG plus the full `FailureCaseMetadata` JSON lands at
`/sdcard/Android/data/io.kitware.shitspotter/files/failure_cases/<ts>/`.

The HUD numbers are computed in
[`composeApp/src/androidMain/.../CameraAnalysisLoop.kt`](composeApp/src/androidMain/kotlin/io/kitware/shitspotter/android/CameraAnalysisLoop.kt)
and pushed into the shared
[`AppState`](composeApp/src/commonMain/kotlin/io/kitware/shitspotter/core/AppState.kt)
on every analyzed frame. `dropped` increments while `Pause` is active.
