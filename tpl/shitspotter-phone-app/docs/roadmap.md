# ScatSpotter / ShitSpotter Phone App — Roadmap

**Status as of 2026-05:** Working Android alpha on Pixel 5 via sideload. Gesture/zoom/photo-review
features are functional. Architecture is solid. Not yet Play Store or App Store ready.

Two independent reviews converge on the same verdict: **professional internal alpha**. The abstractions
(`DetectorBackend`, `FrameSource`, `ModelSpec`, shared core, desktop harness, platform adapters) are
already in the right shape. What remains is the release/operations layer, not a structural rewrite.

---

## Phase 0 — Immediate (before any public distribution)

These are blockers. None of them require large refactors.

### 0.1 Production signing

Release builds currently use the debug signing config
(`composeApp/build.gradle.kts`, `signingConfig = signingConfigs.getByName("debug")`).
Losing the debug key = losing the ability to update the app on any store.

- Generate a production keystore stored **outside the repo**.
- Read it from environment variables or `local.properties` (gitignored).
- Add `applicationIdSuffix = ".debug"` to the debug variant so a signed release and a dev
  build can coexist on the same phone.
- Document the keystore/env setup in `001_build_run_validate.md`.

### 0.2 Target SDK 35

Google Play has required new apps and updates to target Android 15 / API 35 since August 31, 2025.

```kotlin
compileSdk = 35
targetSdk  = 35
```

Remove `android.suppressUnsupportedCompileSdk=34` from `gradle.properties` once done.

### 0.3 Fix location permission timing

The manifest declares `ACCESS_FINE_LOCATION` + `ACCESS_COARSE_LOCATION`, and `MainActivity`
requests them immediately after camera permission — even though the default metadata mode
is `NO_GPS`. This hurts Play Store review posture (permissions must be contextually justified).

Fix: request location **only** when the user enables "Include GPS location" in settings.
If denied, keep `MetadataMode.NO_GPS` and show a one-line explanation.
Never ask for location on first launch.

### 0.4 Remove hardcoded recipient email

`AppState` initializes a personal email as the default share recipient. In a public build this
looks unpolished and raises privacy review questions even though no upload is automatic.
Make the default blank; let the share sheet handle recipient selection.

### 0.5 Privacy policy

CAMERA + LOCATION permissions require a privacy policy URL in Play Console.
Content is simple given the offline-only story:

- What is collected: photos + optional GPS coordinates, stored locally.
- What leaves the device: nothing except what the user explicitly shares via the OS share sheet.
- No `INTERNET` permission is declared; no background uploads exist.

A static page (GitHub Pages or a simple hosted file) is sufficient.

---

## Phase 1 — Near-term (before Play internal testing or TestFlight)

### 1.1 AAB release path + CI lane

Play Store requires Android App Bundles for new app submissions; APKs are for sideload/internal
testing. Both are useful; keep both.

Add explicit Gradle tasks:

```bash
./gradlew :composeApp:bundleRelease    # Play Store upload
./gradlew :composeApp:assembleRelease  # sideload / AltStore / F-Droid
```

CI lane (runs without a phone):

```bash
./gradlew :composeApp:desktopTest :composeApp:lintRelease :composeApp:assembleRelease :composeApp:bundleRelease
```

Optional/manual device lane:

```bash
./gradlew :composeApp:connectedAndroidTest
scripts/install_to_phone.sh release --run
```

### 1.2 Version code strategy

`versionCode = 1` and `versionName = "0.1.0"` are hardcoded. versionCode must strictly increase
with every store upload. Drive these from a property file or git tag:

```kotlin
// gradle.properties: APP_VERSION_CODE=1, APP_VERSION_NAME=0.1.0
versionCode = providers.gradleProperty("APP_VERSION_CODE").get().toInt()
versionName = providers.gradleProperty("APP_VERSION_NAME").get()
```

### 1.3 Keep screen on during camera view

The viewfinder dims and locks while the user is walking. Add to the camera composable:

```kotlin
// Modifier on the camera surface Box:
.keepScreenOn()  // androidx.compose.ui.platform
```

Or set `WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON` in `MainActivity.onCreate`.
Clear it when entering review mode so the screen can sleep normally there.

### 1.4 Data Safety form

Fill out the Play Console Data Safety declaration. Given the current design:

- No data shared with third parties.
- No data collected (no network, no analytics SDK).
- User can delete all data by clearing app storage.

This is a short form and a real differentiator vs. apps with trackers.

---

## Phase 2 — Battery and Performance

The hot path currently makes several per-frame heap allocations: RGBA copy, RGBA→RGB strip,
letterbox, RGB→float array, and `OnnxTensor` wrapping. This causes GC pressure and increased
battery use.

### 2.1 Reuse hot-path buffers

Create a `FramePreprocessor` that owns fixed-size scratch buffers per active model. Reallocate
only on model switch or camera resolution change. Per-frame work fills existing buffers.

```kotlin
class FramePreprocessor(modelInputW: Int, modelInputH: Int) {
    private val rgbBuf  = ByteArray(modelInputW * modelInputH * 3)
    private val floatBuf = FloatArray(modelInputW * modelInputH * 3)
    // ...
}
```

### 2.2 YUV → tensor shortcut (Android)

CameraX currently outputs `RGBA_8888`. For camera ML pipelines, `YUV_420_888` is often cheaper
because one channel (Y) is already a contiguous plane and many ONNX models accept it directly.
Keep the pure-Kotlin RGBA path as the reference/desktop implementation; let the Android adapter
use a platform-optimized YUV path.

### 2.3 Power mode setting

The model registry already records speed hints (e.g. "11 FPS", "3 FPS" on Pixel 5 with NNAPI).
Expose a user-visible power mode that controls target inference rate:

| Mode        | Behavior                              |
|-------------|---------------------------------------|
| Battery     | 320 px model, target 3–5 FPS          |
| Balanced    | 416 px model, target 8–10 FPS         |
| Performance | full selected model, no throttle      |

The existing `CameraAnalysisLoop` already has `STRATEGY_KEEP_ONLY_LATEST`; add a frame-rate
gate and wire it to the selected power mode.

---

## Phase 3 — Model Publishing Pipeline

The current `ModelRegistry` is a good development tool but requires a Kotlin code edit for every
new model. For frequent model updates, decouple models from app releases.

### 3.1 JSON model manifest format

```json
{
  "modelId": "shitspotter-deimv2_pico-h320w320",
  "displayName": "DEIMv2-Pico 320×320",
  "format": "ONNX",
  "modelFile": "deimv2_pico_h320_w320.onnx",
  "sha256": "...",
  "inputWidth": 320,
  "inputHeight": 320,
  "layout": "NCHW",
  "normalization": { "scale": 0.00392156862 },
  "postprocess": { "type": "DEIMV2" },
  "metrics": { "ap50": 0.265, "pixel5NnapiMs": 89 }
}
```

### 3.2 Model loading priority chain

1. Bundled default model pack (always present; guarantees feature works at review time).
2. App-private `files/models/<model-id>/manifest.json` (downloaded or pushed via adb).
3. User-imported model pack via Android document picker / iOS Files app.

`AndroidModelLoader` already checks external files, cache, and assets in order — extend that
logic to parse manifests and verify SHA-256 before loading.

### 3.3 Sideload model script

Update `sync_failure_cases.sh` / add a `push_model.sh`:

```bash
adb push model.onnx /sdcard/Android/data/io.github.erotemic.shitspotter/files/models/
adb push manifest.json /sdcard/Android/data/io.github.erotemic.shitspotter/files/models/
```

---

## Phase 4 — Architecture and Maintainability

### 4.1 Module split

A single `composeApp` is fine for internal development. For shipping, split into Gradle modules:

```
:core              # geometry, model spec, telemetry, serialization
:preprocessing     # reference CPU preprocessing (platform-independent)
:ui                # shared Compose UI (AppScreen, HUD, Overlay, PhotoViewer)
:androidApp        # CameraX, ORT Android, PhotoStore, permissions, MainActivity
:desktopHarness    # compare CLI, desktop harness, validation tooling
```

Benefits: faster incremental builds, cleaner ownership, iOS target more natural, release APK
no longer entangled with desktop harness code.

### 4.2 Decompose MainActivity

`MainActivity` currently owns permissions, settings, backend selection, photo capture, email
sharing, location, review navigation, and lifecycle (lines 55–401). Extract:

```kotlin
PermissionController   // camera + location request flow
LocationProvider       // last-known location with permission guard
CaptureController      // photo capture, shutter sound, metadata assembly
ShareController        // FileProvider URIs, email intent construction
AndroidBackendViewModel // settings persistence, model switching
```

This makes each piece independently testable and Play policy review easier.

### 4.3 Real instrumented tests

`PlaceholderInstrumentedTest.kt` is a stub. Add at minimum:

- Smoke test: app launches, camera permission dialog appears.
- Settings round-trip: write + read settings store, values match.
- Model loader: given a file at the expected path, `AndroidModelLoader` returns a backend.

---

## Phase 5 — iOS

The iOS path is architecturally plausible (KMP source sets, `expect`/`actual` scaffolding) but
not a real app yet. The right next step is building a real shell before any App Store or
TestFlight work.

### 5.1 What exists

- `iosMain` source set with `IosActuals.kt` (only fills `nowMonoMs` and `BuildInfo`).
- `iosMain/.../ImageLoader.kt` returns `null`.
- `iosMain/.../BackHandler.kt` is a stub.
- No `iosApp/` Xcode project.
- No AVFoundation camera surface.
- No iOS inference backend (no Core ML / ORT CoreML / LiteRT delegate integration).
- No iOS photo store, settings store, or permissions.

### 5.2 Build the shell

```
iosApp/
  ScatSpotter.xcodeproj
  ScatSpotter/
    Info.plist                      # NSCameraUsageDescription, NSLocationWhenInUseUsageDescription,
                                    # NSPhotoLibraryAddUsageDescription
    PrivacyInfo.xcprivacy           # required for App Store submissions
    ContentView.swift               # hosts ComposeUIViewController
    IosCameraSurface.kt             # AVFoundation capture session
    IosBackendManager.kt            # ORT CoreML / Core ML backend
    IosPhotoStore.kt
    IosSettingsStore.kt
```

For inference, avoid routing full frames through shared Kotlin. Use AVFoundation to capture,
pass to ORT CoreML or Core ML, return compact `Detection` list to shared UI code. This matches
the existing `DetectorBackend` contract.

### 5.3 Privacy manifest

Apple requires `PrivacyInfo.xcprivacy` for all new app submissions. Declare camera and location
as required-reason APIs. Plan this from the start; retrofitting is harder.

### 5.4 iOS sideloading reality check

iOS sideloading is more constrained than Android:

| Path | Notes |
|------|-------|
| TestFlight | Normal Apple Developer Program; simplest for beta testing |
| AltStore Classic | `.ipa` expires after 7 days; limited to 3 sideloaded apps |
| AltStore PAL / alternative marketplaces | EU/Japan only; still requires App Store Connect account |
| App Store | Standard $99/year path; widest reach |

Don't treat iOS sideloading as equivalent to Android APK distribution — it's more operationally
expensive. TestFlight is the correct "before App Store" path.

---

## Distribution Targets Summary

| Target | Status | Blocker |
|--------|--------|---------|
| Android sideload (APK via adb) | **Working** | None |
| Android internal testing (Play Console) | Near-ready | Phase 0 items |
| Android Play Store | Not ready | Phase 0 + Phase 1 |
| Android F-Droid | Feasible | Reproducible build setup |
| iOS TestFlight | Not started | Phase 5 shell |
| iOS App Store | Not started | Phase 5 + App Review compliance |
| iOS AltStore Classic | Feasible once iosApp exists | 7-day expiry, 3-app limit |
| iOS AltStore PAL | EU/Japan only | Phase 5 + Alt marketplace setup |

---

## Naming Reference

| Context | Name |
|---------|------|
| Home screen, store listing, permission dialogs, share subjects | **ScatSpotter** |
| Code, log tags, variable names, git history, this doc | **ShitSpotter** |
| Android package ID (not user-visible) | `io.github.erotemic.shitspotter` |
| iOS bundle ID (when created) | `io.github.erotemic.shitspotter` |
