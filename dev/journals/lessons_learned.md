# Lessons learned

Distilled postmortems per [`../AGENT_BENCHMARK_DISCIPLINE.md`](../AGENT_BENCHMARK_DISCIPLINE.md)
§Journals. Less formal than `benchmark-candidates/`; if an entry
crystallises into a clear invariant, also create a benchmark
candidate.

---

## 2026-05-10: Phone-app v2 scaffold — five hard invariants in one run

### Trigger

User-launched autonomous run with a 2-hour budget to scaffold a
KMP+Compose replacement for the .NET MAUI app at `tpl/poopdetector/`.
GOAL.md target: Milestone 0 (decision) minimum, Milestone 1 (skeleton)
soft success, Milestone 2 (real model backend) 90%+ true success.

### Symptoms

Five categories of subtle failure that surfaced during the run:

1. **ONNX shape mismatch** — `ai.onnxruntime.OrtException: Got
   invalid dimensions for input: images, index: 2 Got: 640 Expected:
   416` from deep inside `session.run`, not at construction time.
2. **Overlay misalignment** — Detection boxes rendered in a different
   region of the screen than the live camera image.
3. **Rotation drift** — On Pixel 5 portrait-locked, the camera buffer
   arrives at `rotationDegrees=90`, but the model produces detections
   in buffer (landscape) coordinates while the preview is rotated to
   portrait.
4. **YOLOX-nano OOD over-fire** — Stock dog photo produces a 0.757
   confidence "poop" detection.
5. **Bloated APK** — First `assembleDebug` produced 82 MB.

### Root cause

1. ORT defers tensor-shape validation until first `run()`. Our
   backend was constructed with a wrong-sized `ModelSpec` and the
   error only surfaced on the first frame.
2. CameraX `PreviewView` defaults to `FILL_CENTER`; our shared
   `DetectionOverlay` defaulted to a min-scale letterbox
   (effectively `FIT_CENTER`). Two coordinate spaces, same `Box`.
3. CameraX delivers buffers in native sensor orientation and
   reports `rotationDegrees` separately. PreviewView handles the
   display rotation internally; the overlay didn't.
4. The shipped YOLOX-nano was trained on pre-cropped patches; full-
   image OOD inputs produce high-confidence garbage.
5. AGP defaults include `armeabi-v7a`, `arm64-v8a`, `x86`, `x86_64`
   for every native `.so`. ONNX Runtime ships ~20 MB per ABI.

### Fix

1. Added `validateInputShape()` to both ONNX backends that reads
   `session.inputInfo[name].info` and asserts H/W match the spec,
   with a Kotlin-side error referencing the `modelId`.
   **Fix SHA:** `21a3b43` (pre-error at `92edfe4`).
2. Added `OverlayScaleMode` enum + `CameraSurface.overlayScaleMode`
   property. Android surface declares `FILL_CENTER`, desktop still-
   image declares `FIT_CENTER`.
   **Fix SHA:** `1eba4b8` (pre-error at `40197ac`).
3. Added pure-Kotlin `BoundingBox.rotated(degrees, w, h)` and rotated
   every detection plus the frame W/H before pushing into AppState.
   **Fix SHA:** `feca293` (pre-error at `e82ccc3`).
4. Documented the OOD behaviour in `ModelSpec.notes` and in
   `docs/004_kotlin_python_parity.md`. Did not change the detection
   pipeline.
   **Fix SHAs:** `fe1b1c8` (docs) + `eb85008` (ModelSpec.notes)
   (pre-error at `92edfe4`).
5. Set `ndk.abiFilters = setOf("arm64-v8a")` for Pixel 5. APK shrank
   82 MB → 29 MB.
   **Fix SHA:** `8e4f892` (pre-error at `e82ccc3`).

### Durable lesson

Every cross-stack ML deployment task touches at least *three* fragile
boundary layers that don't show up as compile errors:

- **Shape boundary** between ModelSpec and the real ONNX file (fails
  late inside native code).
- **Coordinate boundary** between the buffer the detector sees and
  the display the user sees (manifests as silent visual offsets, not
  as exceptions).
- **Training-data boundary** between what the model was trained on and
  what the camera ingests (silent over- or under-firing).

Make the failure mode loud at the earliest possible point in each
boundary, and document the OOD-behaviour story so the next agent
doesn't try to "fix" it as a regression.

A fourth boundary — **packaging** — is also worth front-loading.
AGP's "ship every ABI" default is almost always wrong for a known
target.

### Candidate follow-up

Each of the five symptoms is a benchmark candidate in
`dev/benchmark-candidates/app-deployment-questions.md`.

Cross-references to consider for further candidates:

- **TFLite parity** — when LiteRT gets wired up, the same five
  boundary layers will produce *different* failure modes. Re-run
  the candidate template against LiteRT specifically.
- **Front camera vs back camera** — the v2 app has a `useFrontCamera`
  toggle. Front cameras on most phones report a *mirrored* preview
  in PreviewView but unmirrored buffers in ImageAnalysis. The
  overlay alignment story is incomplete until that case is
  validated on a real phone.
- **Quantised models** — the registered `*-float16.onnx` variant has
  a different output dtype and currently isn't loaded. When it is,
  the YOLOX postprocess will need a dtype-aware path.

### Validation status

What ran on the VM (Linux desktop, JVM CPU):

- 122 unit tests across 28 files, all green.
- ONNX smoke test against the real
  `tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx`.
- `scripts/python_reference_compare.py` against the same model +
  `dog.jpg`; top-3 detections within ~0.01 of Kotlin output.
- `compare` CLI against three registered models on the same image.
- Both `assembleDebug` (29 MB) and `assembleRelease` (20 MB).

What did **not** run on the VM (requires phone):

- Live FPS measurement, NNAPI delegate verification, preview
  smoothness, thermal/battery behaviour, real-world over-fire
  observation, front-camera mirroring, on-device failure-case
  capture.

Each is documented as such in
`dev/journals/2026-05-10_phone_app_kmp_scaffold.md` and
`tpl/shitspotter-phone-app/docs/001_build_run_validate.md`.
