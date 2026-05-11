# Lessons learned

Distilled postmortems per [`../AGENT_BENCHMARK_DISCIPLINE.md`](../AGENT_BENCHMARK_DISCIPLINE.md)
§Journals. Less formal than `benchmark-candidates/`; if an entry
crystallises into a clear invariant, also create a benchmark
candidate.

---

## 2026-05-11: mobile_app_training_v4 — four cross-library API failures the agent could not have caught

### Trigger

Scaffolded `experiments/mobile_app_training_v4/` from a VM with no GPU and
no shitspotter Python env on PATH. All design + syntax + import-check
work passed in the VM. The user then ran `run_all.sh` on the host and
hit four runtime failures in sequence — each one a wrapper-API misuse
that compiled fine and parsed fine but exploded at first invocation.

### Symptoms

1. `gdown --fuzzy` →
   `__main__.py: error: unrecognized arguments: --fuzzy`
2. `kwimage.imresize(image, (W, H), interpolation='area')` →
   `cv2.error: Failed to allocate 44947419955200 bytes` on the first
   image. The (W, H) tuple was interpreted as the *scale* parameter,
   not `dsize`.
3. `kwimage.imwrite(fpath, image, imwrite_params=[('JPEG_QUALITY', 90)])` →
   `cv2.error: imwrite() got an unexpected keyword argument 'imwrite_params'`.
   kwimage forwards `**kwargs` straight to `cv2.imwrite`, which expects
   a flat `params=[INT_FLAG, INT_VALUE, ...]` list, not a name=value
   kwarg, and definitely not whatever the kwimage-specific name was.
4. `gdown` failure-mode caching: when Drive returned an HTML "quota
   exceeded" stub, the saved `.pth` was a few KB. The next run's
   `if [ -f "$dst" ]` short-circuit happily reused the bogus file as
   "downloaded".

### Root cause

All four are the same shape: **shelling out to or wrapping a
third-party API without verifying the call signature in the version
actually installed.** The agent had no way to test against the real
runtime — a hosted VM with no GPU and no shitspotter env can't
exercise gdown, kwimage, or DEIMv2 — so every API call became a
bet against an assumed signature.

The specific assumptions that broke:

* `gdown --fuzzy` was canonical in 4.x/5.x; gdown 6.0.0 (released
  2026) dropped it because the URL parser now handles every form
  natively. Old example commands transplanted into v4 docs without
  re-checking the upstream changelog.
* `kwimage.imresize`'s second positional is `scale`, not `dsize`.
  The call read like English ("imresize this image to (1280, 960)")
  and the misuse wasn't a type error — it was a value semantics error
  that compiled fine and only surfaced as a multi-TB allocation.
* `kwimage.imwrite` forwards `**kwargs` to `cv2.imwrite`. The
  documentation for the friendly knob (JPEG quality) lives in cv2's
  C++ docs. The pythonic-looking `imwrite_params=[('NAME', value)]`
  form was hallucination, neither kwimage's API nor cv2's.
* `gdown` writes whatever Drive returns to the destination path,
  including HTML error pages, with exit code 0. The "did the file
  download?" question is *not* answered by `[ -f "$dst" ]`.

### Fix

* `00_setup.sh`: drop `--fuzzy`, pass the bare file ID (works in
  every gdown version), and require the saved file to be ≥ 1 MiB
  before treating it as cached. SHA `8e905fb`.
* `tile_kwcoco.py:_resize_with_long_side`: pass `dsize=` explicitly
  to `kwimage.imresize`. SHA `985b5d6`.
* `tile_kwcoco.py:_imwrite`: drop the spurious `imwrite_params`
  kwarg; build the cv2-style flat `params=[cv2.IMWRITE_JPEG_QUALITY,
  q]` list when JPEG, fall through to defaults otherwise. SHA below.

### Durable lesson

When writing wrapper code from inside an environment that cannot
run the wrapped library, **the call site is at risk for every
positional, every kwarg name, every CLI flag**. Mitigations, in
descending order of value:

1. **Prefer keyword arguments at every wrapper call site, even when
   the positional form looks self-explanatory.** `kwimage.imresize(
   img, dsize=(W, H))` makes the next bug-of-its-kind a `TypeError`
   instead of a 45 TB allocation.
2. **Treat external-CLI defaults as version-bound.** Pin the version
   in setup, or re-read the upstream `--help` before quoting flags
   from memory. `--fuzzy` was load-bearing for two years; that's how
   long the agent's mental model can be wrong.
3. **Never trust file-presence as success for a downloaded artifact.**
   Min-size guard (this fix), or a hash check (better), or both.
4. **The first run on the real host is the test suite.** Plan for it.
   The four bugs above are exactly what a 5-minute smoke run would
   have caught — the v4 `V4_RUN_ALL_SMOKE=1` mode now exists for
   that reason. Use it before the full sweep.

### Candidate follow-up

* If a similar wrapper cluster appears again, promote
  "wrapper API misuse" to a benchmark candidate in
  `dev/benchmark-candidates/`. For now the four examples above are
  variations on the same theme; one entry covers it.
* The kwimage-imresize footgun in particular is generic — any agent
  writing wrappers around kwimage should default to `dsize=` /
  `scale=` keyword form rather than the positional shortcut.

### Validation status

After all four fixes, `run_all.sh` re-run by the user.
Step 0 reused the (real, > 1 MiB) DEIMv2 weights and step 1
proceeded past the first image without crashing. Subsequent steps
were not yet exercised at the time of writing this entry.

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
