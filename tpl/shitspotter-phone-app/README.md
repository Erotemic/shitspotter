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
  docs/
    000_stack_decision.md      # Milestone 0 — write this first
  composeApp/
    src/commonMain/            # shared UI state, model registry, result types
    src/androidMain/           # CameraX, NNAPI/CoreML execution providers, etc.
    src/desktopMain/           # Linux still-image/video harness
    src/iosMain/               # scaffolding only — cannot build from this VM
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
