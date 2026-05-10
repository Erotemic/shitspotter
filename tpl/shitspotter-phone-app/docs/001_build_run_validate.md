# 001 — Build, run, and validate

This is the operator-facing checklist for building the prototype, running it
on the Linux desktop harness, and sideloading it onto a Pixel 5. It assumes
you've already read [`README.md`](../README.md) and have sourced the
toolchain env file.

```bash
source /data/tmp/shitspotter-app-toolchain/env.sh
cd /home/joncrall/code/shitspotter/tpl/shitspotter-phone-app
```

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

The first build downloads multi-GB of Android Gradle Plugin, AndroidX,
Compose, CameraX, and ONNX Runtime artifacts into `~/.gradle/caches/`.
Subsequent builds reuse them.

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

---

## 6. Sideload to a Pixel 5

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

## 7. What the user should report back

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

## 8. Known limitations on this VM

- No webcam → no live camera test for the desktop harness; still images only.
- No USB passthrough → no on-device APK install from the VM.
- No Android emulator hardware acceleration → emulator-based smoke tests
  are slow but possible if you really need them.
- No macOS/Xcode → iOS source set is scaffolding only; no `linkDebug*`
  targets are configured.

These limits are deliberate and documented in
[`docs/000_stack_decision.md`](000_stack_decision.md) §"Risks / open
items". They do not block Milestone 0/1/2 progress.
