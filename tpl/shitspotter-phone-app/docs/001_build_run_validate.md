# 001 — Build, run, and validate

This is the operator-facing checklist for building the prototype, running it
on the Linux desktop harness, and sideloading it onto a Pixel 5.

---

## 0. Toolchain setup — choose your machine

### Dev VM (the shitspotter VM with `/data/tmp/`)

The full toolchain (Temurin JDK 17, Android SDK, adb) lives under
`/data/tmp/shitspotter-app-toolchain/`. **Source this file in every shell
that runs a build:**

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
# sets JAVA_HOME, ANDROID_HOME, and prepends bin dirs to PATH
```

Verify with:

```bash
java -version    # expect Temurin 17.0.12+
javac -version   # must match
adb --version
```

If anything is missing, re-run the installer (idempotent, ~1 GB):

```bash
bash install_toolchain.sh          # JDK 17 + Android SDK
bash install_toolchain.sh --check  # verify existing install
```

### Local workstation (e.g. `toothbrush`)

The `env.sh` above is VM-specific. On your own machine install the
prerequisites directly, then set `JAVA_HOME` before running Gradle:

```bash
# Ubuntu/Debian — install the full JDK (not just the JRE):
sudo apt install openjdk-21-jdk android-tools-adb

# Point JAVA_HOME at the JDK that provides javac:
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
```

`ANDROID_HOME` must also be set if you want to build Android targets.
If you have Android Studio, its SDK is usually at `~/Android/Sdk`:

```bash
export ANDROID_HOME="$HOME/Android/Sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
```

Add both exports to `~/.bashrc` / `~/.zshrc` so you don't have to repeat
them each session.

**After changing `JAVA_HOME` or installing a new JDK**, kill any running
Gradle daemon — it caches the old toolchain detection:

```bash
./gradlew --stop
./gradlew clean :composeApp:assembleRelease
```

> **Why the JDK, not the JRE?**
> Ubuntu ships `openjdk-21-jre` and `openjdk-21-jdk` as separate packages.
> The JRE directory (`/usr/lib/jvm/java-21-openjdk-amd64`) lacks `javac`,
> and Gradle's toolchain resolver validates capability `JAVA_COMPILER` by
> checking that directory — not just `PATH`. Installing `openjdk-21-jdk`
> adds `javac` to that same directory. The Gradle daemon must be restarted
> after the install because it caches the capability scan.

---

---

## 1. Sanity check the toolchain

```bash
java -version           # expect Eclipse Temurin 17.0.12+
adb --version           # expect platform-tools 37.0.0+
sdkmanager --version    # expect 12.0
./gradlew --version     # confirms the wrapper resolves
```

---

## 2. Run shared-core unit tests (fastest signal)

This compiles only `commonMain` + `desktopMain` and runs the JUnit-style
tests in `composeApp/src/commonTest/`:

```bash
./gradlew :composeApp:desktopTest --console=plain
```

Expected: `BUILD SUCCESSFUL` and a green test run for `GeometryTest` and
`PreprocessingTest`. These cover IoU, NMS, letterbox round-trip, YOLOX
postprocess filtering, NCHW vs NHWC layout, RGB/BGR swap.

---

## 3. Run the Linux desktop harness (Compose for Desktop)

The desktop harness is the Linux-native test path. It loads the same
shared pipeline used on Android, but reads still images from disk
instead of CameraX frames. By default it uses the stub detector
(Milestone 1) and any image you point it at. To use the real ONNX
backend, see §5.

```bash
# stub detector with a still image
./gradlew :composeApp:run --args="--image=/home/joncrall/code/shitspotter/tpl/poop_models/README_assets/example.jpg"
```

If you don't have an image handy, omit `--image=` to see the empty
preview UI; the HUD still updates with stub telemetry and overlay math
runs the same code path as on Android.

You can also pass the image via env var:

```bash
SHITSPOTTER_DESKTOP_IMAGE=/path/to/image.jpg ./gradlew :composeApp:run
```

---

## 4. Build the Android debug APK

```bash
./gradlew :composeApp:assembleDebug --console=plain
```

If the build succeeds, the APK is at:

```text
composeApp/build/outputs/apk/debug/composeApp-debug.apk
```

Roughly 29 MB on the current configuration (arm64-v8a only, ONNX
Runtime native libs included). The first build downloads multi-GB of
Android Gradle Plugin, AndroidX, Compose, CameraX, and ONNX Runtime
artifacts into `~/.gradle/caches/`. Subsequent builds reuse them.

For a smaller APK (~20 MB) with R8 minification and resource shrinking,
build the release variant:

```bash
./gradlew :composeApp:assembleRelease
# → composeApp/build/outputs/apk/release/composeApp-release.apk
```

The current release config reuses the debug signing key — fine for
sideloading but **not** suitable for Play Store distribution. Replace
the signingConfig with a real keystore + secret-managed credentials
before any public release; see [`docs/003_known_limitations.md`](003_known_limitations.md) #5.

---

## 5. Plug in the YOLOX-nano poop model (Milestone 2)

The model file is **not committed** anywhere in `tpl/shitspotter-phone-app/`.
Three placement options, in order of agent-friendliness:

### 5a. Bundle into the APK at build time (recommended for sideload tests)

```bash
mkdir -p composeApp/src/androidMain/assets
cp ../poop_models/yolox_nano_poop_cropped_only_best.onnx \
   composeApp/src/androidMain/assets/yolox_nano_poop_cropped_only_best.onnx
./gradlew :composeApp:assembleDebug
```

`composeApp/src/androidMain/assets/` is gitignored via the existing
`*.onnx` rule, so the binary stays out of git. The runtime opens the
asset via `AssetManager` and writes it to the app's cache directory on
first launch.

### 5b. Push to the device manually

```bash
adb push ../poop_models/yolox_nano_poop_cropped_only_best.onnx \
   /sdcard/Android/data/io.kitware.shitspotter/files/models/
```

Then launch the app — the model loader will check
`<external-files-dir>/models/` before falling back to bundled assets.

### 5c. Use it from the desktop harness

```bash
./gradlew :composeApp:run --args="--image=<path> \
    --model=/home/joncrall/code/shitspotter/tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx \
    --model-id=yolox-nano-poop-cropped-v1"
```

The desktop harness uses ONNX Runtime JVM (CPU only).

### 5d. Registered models and where to put them

The chips in the app correspond to entries in `ModelRegistry.all`. To
make a chip do real inference instead of fall back to the stub, push
the matching ONNX file under the exact `modelFile` name into the
device's external-files models dir:

```bash
DEST=/sdcard/Android/data/io.kitware.shitspotter/files/models/
adb shell mkdir -p $DEST

# YOLOX-nano poop (416x416)
adb push ../poop_models/yolox_nano_poop_cropped_only_best.onnx $DEST

# Simple v3 run v06 (YOLOv9, 640x640) — note the file rename
adb push '/data/joncrall/dvc-repos/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.onnx' \
    $DEST/shitspotter-simple-v3-run-v06.onnx

# Custom v5 (640x640)
adb push ../poop_models/shitspotter-custom-v5-epoch_115.onnx $DEST

# Custom v2 (640x640)
adb push ../poop_models/shitspotter_custom_v2_epoch126.onnx $DEST

adb shell ls -lh $DEST
```

The destination filename **must** match the `modelFile` field on the
corresponding `ModelSpec` (see
[`composeApp/src/commonMain/.../ModelSpec.kt`](../composeApp/src/commonMain/kotlin/io/kitware/shitspotter/core/ModelSpec.kt));
otherwise the chip falls back to stub and the HUD's chip shows the ⚠
indicator added in commit 20aa1ae.

After pushing, just tap the chip again — `setActive` re-checks the
external dir on every tap.

---

## 6. iOS path (macOS host only)

iOS is **unvalidated** from this Linux VM. The KMP source set + actuals
exist; the Gradle target wiring activates automatically when the build
detects a macOS host.

On a macOS host with Xcode installed:

```bash
cd tpl/shitspotter-phone-app
./gradlew :composeApp:linkDebugFrameworkIosArm64        # framework only
./gradlew :composeApp:embedAndSignAppleFrameworkForXcode # for Xcode
open iosApp/iosApp.xcodeproj    # NOT YET CREATED — see docs/003 #7
```

To force iOS target configuration on Linux (e.g. to type-check
`iosMain/`), pass `-Pssp.enableIosTargets=true`. This will trigger
Kotlin/Native toolchain download and is not part of the normal Linux
dev cycle.

The actual iOS app shell (Xcode project, AVFoundation camera bridge,
Core ML / ORT-CoreML inference path) is **not** scaffolded — see
`docs/003_known_limitations.md` items #7 and #11.

---

## 7. Sideload to a Pixel 5

The VM has no USB passthrough. Run these from your workstation:

```bash
# on workstation
adb devices                                 # confirm Pixel 5 is connected
adb install -r /path/to/composeApp-debug.apk
adb logcat -s "ShitSpotter.AnalysisLoop:V" "ShitSpotter.Failure:V"
```

After install, launch the **ShitSpotter** app from the launcher. Grant
camera permission on first run. The HUD in the top-left shows FPS,
detection count, and per-stage latency. Tap **Save failure** to write a
failure-case report under
`/sdcard/Android/data/io.kitware.shitspotter/files/failure_cases/`.

To pull failure cases back to the workstation:

```bash
adb pull /sdcard/Android/data/io.kitware.shitspotter/files/failure_cases/ \
   ./pulled_failure_cases/
```

---

## 8. What the user should report back

After running on a Pixel 5, please share:

```text
- adb shell getprop ro.build.fingerprint
- adb shell dumpsys package io.kitware.shitspotter | grep versionName
- HUD readings (FPS, inference ms, delegate, model id)
- whether preview stays smooth when inference is slow
- contents of any failure_cases/<timestamp>/metadata.json
- any unexpected logcat lines tagged ShitSpotter.*
```

---

## 9. Troubleshooting

### `Toolchain installation … does not provide the required capabilities: [JAVA_COMPILER]`

Gradle's JVM toolchain resolver checked a specific directory and found no
`javac` there. Three likely causes:

| Cause | Fix |
|---|---|
| Only the JRE is installed (`openjdk-21-jre`, not `openjdk-21-jdk`) | `sudo apt install openjdk-21-jdk` |
| `JAVA_HOME` not set — Gradle picked up the wrong directory | `export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))` |
| Gradle daemon cached the old (pre-JDK-install) detection | `./gradlew --stop` then retry |
| On the VM, forgot to source `env.sh` | `source /data/tmp/shitspotter-app-toolchain/env.sh` |

All four can compound. The safe reset sequence on a local workstation:

```bash
sudo apt install openjdk-21-jdk                     # ensure JDK present
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
./gradlew --stop                                     # kill stale daemon
./gradlew clean :composeApp:assembleRelease
```

### `ANDROID_HOME not set` / aapt2 not found

```bash
export ANDROID_HOME="$HOME/Android/Sdk"             # adjust to your SDK path
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
```

### Build hangs downloading Gradle distribution

The Gradle wrapper downloads `gradle-8.10.2-bin.zip` on first run (~130 MB).
It caches under `~/.gradle/wrapper/dists/`. If the download hangs, kill it
and retry — the partial download is cleaned up automatically.

---

## 10. Known limitations on this VM

- No webcam → no live camera test for the desktop harness; still images only.
- No USB passthrough → no on-device APK install from the VM.
- No Android emulator hardware acceleration → emulator-based smoke tests
  are slow but possible if you really need them.
- No macOS/Xcode → iOS source set is scaffolding only; no `linkDebug*`
  targets are configured.

These limits are deliberate and documented in
[`docs/000_stack_decision.md`](000_stack_decision.md) §"Risks / open
items". They do not block Milestone 0/1/2 progress.
